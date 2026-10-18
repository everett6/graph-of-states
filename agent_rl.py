"""Multi-turn tool use inside GRPO rollouts (agentic RL).

Each sampled completion may call the Python sandbox with <run>...</run> (same protocol as agent_loop.py).
Generation stops at </run>, the code runs, and the tool result is spliced into the token stream as a user
turn rendered by the model's own chat template. Tool-result tokens get env_mask 0, so TRL excludes them
from the policy loss; the model is trained only on tokens it generated. The reward sees the final
assistant segment (`agent_final`), so text printed by a tool cannot be graded as the answer.
"""
from agent_loop import RUN, tool_call, tool_message

SENTINEL_ANSWER = 'GOS_SENTINEL_ANSWER'
SENTINEL_OUTPUT = 'GOS_SENTINEL_OUTPUT'


def env_template(tokenizer):
    """Text between an assistant turn's content and the next assistant turn, with an `{output}` slot,
    taken from the tokenizer's chat template (so every model gets its own turn markers)."""
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'task'}, {'role': 'assistant', 'content': SENTINEL_ANSWER},
         {'role': 'user', 'content': SENTINEL_OUTPUT}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    between = rendered.split(SENTINEL_ANSWER, 1)[1]
    if between.count(SENTINEL_OUTPUT) != 1:
        raise ValueError('Chat template does not render a follow-up user turn')
    return between.replace('{', '{{').replace('}', '}}').replace(SENTINEL_OUTPUT, '{output}')


def agent_turns(prompt_ids, generate, run_tool, render_env, decode, max_tools, budget, stop_ids):
    """One agentic completion. generate(ids, max_new) -> new model token ids; run_tool(code) -> sandbox
    outcome dict; render_env(message) -> token ids; decode(ids) -> text.
    Returns (completion_ids, env_mask, final assistant text, tool calls)."""
    completion, mask, calls = [], [], 0
    segment_start = 0
    for turn in range(max_tools + 1):
        room = budget - len(completion)
        if room <= 0:
            break
        new = generate(prompt_ids + completion, room)
        completion += new
        mask += [1] * len(new)
        text = decode(completion[segment_start:])
        code = None
        if turn < max_tools and not (new and new[-1] in stop_ids):
            code, _ = tool_call(text, RUN)
        if code is None:
            break
        calls += 1
        env = render_env(tool_message(run_tool(code), max_tools - turn - 1))
        env = env[:max(0, budget - len(completion))]
        completion += env
        mask += [0] * len(env)
        segment_start = len(completion)
    return completion, mask, decode(completion[segment_start:]), calls


def text_stop(tokenizer, start, text='</run>', window=16):
    """StoppingCriteria: stop once `text` appears in the tokens generated after position `start`
    (decode-based, so it works for any tokenizer)."""
    import torch
    from transformers import StoppingCriteria

    class Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            tail = input_ids[0, max(start, input_ids.shape[1] - window):]
            done = text in tokenizer.decode(tail, skip_special_tokens=True)
            return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)
    return Stop()


def agent_rollout(prompts, trainer):
    """TRL rollout_func: like train_unified.sequential_rollout, plus sandboxed <run> tool calls."""
    import torch
    from gos_tools import execute_python
    config = trainer.gos_config
    model = trainer.accelerator.unwrap_model(trainer.model)
    tokenizer = trainer.processing_class
    template = env_template(tokenizer)
    stop_ids = {tokenizer.eos_token_id, *([tokenizer.convert_tokens_to_ids('<end_of_turn>')]
                                         if '<end_of_turn>' in tokenizer.get_vocab() else [])}

    def generate(ids, room):
        inputs = torch.tensor([ids], device=trainer.accelerator.device)
        with trainer.accelerator.autocast():
            out = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                 use_cache=config.grpo_use_cache, max_new_tokens=room, do_sample=True,
                                 temperature=trainer.temperature, top_k=0, top_p=1.0,
                                 pad_token_id=tokenizer.pad_token_id, logits_to_keep=1,
                                 stopping_criteria=[text_stop(tokenizer, inputs.shape[1])])
        return out[0, inputs.shape[1]:].tolist()

    def run_tool(code):
        return execute_python(code, timeout=10, output_limit=2000)

    def render_env(message):
        return tokenizer.encode(template.format(output=message), add_special_tokens=False)

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    prompt_ids, completion_ids, logprobs, env_mask, finals, tool_calls = [], [], [], [], [], []
    for prompt in prompts:
        if not isinstance(prompt, str):
            raise ValueError('Unified GRPO expects pre-rendered text prompts')
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > config.max_prompt_tokens:
            raise ValueError('GRPO prompt exceeds budget')
        completion, mask, final, calls = agent_turns(ids, generate, run_tool, render_env, decode,
                                                     config.agent_max_tools, config.max_completion_tokens, stop_ids)
        if not completion:  # TRL needs at least one token per completion
            completion, mask = [tokenizer.eos_token_id], [1]
        full = torch.tensor([ids + completion], device=trainer.accelerator.device)
        with trainer.accelerator.autocast():
            chosen, _, _ = trainer._get_per_token_logps_and_entropies(model, full, torch.ones_like(full),
                                                                     len(completion))
        prompt_ids.append(ids)
        completion_ids.append(completion)
        logprobs.append(chosen[0].tolist())
        env_mask.append(mask)
        finals.append(final)
        tool_calls.append(calls)
    return {'prompt_ids': prompt_ids, 'completion_ids': completion_ids, 'logprobs': logprobs,
            'env_mask': env_mask, 'agent_final': finals, 'agent_tool_calls': tool_calls}
