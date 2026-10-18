import math

import pytest

from agent_rl import agent_turns, env_template
from test_unified import tiny_base, tokenizer


def test_env_template_uses_the_chat_template():
    template = env_template(tokenizer())
    assert template == ' [EOS] user: {output} [EOS] assistant: '


def test_agent_turns_masks_tool_output_and_grades_the_final_turn():
    script = [list('think <run>\nprint(1)\n</run>'), list('so \\boxed{1}') + ['EOS']]
    calls = []

    def generate(ids, room):
        return script.pop(0)[:room]

    def run_tool(code):
        calls.append(code)
        return {'success': True, 'timed_out': False, 'stdout': '1\n', 'stderr': ''}

    completion, mask, final, used = agent_turns(
        ['p'], generate, run_tool, lambda message: list('<ENV>'), lambda ids: ''.join(t for t in ids if t != 'EOS'),
        max_tools=2, budget=200, stop_ids={'EOS'})
    assert used == 1 and calls == ['print(1)\n']
    first = len('think <run>\nprint(1)\n</run>')
    assert mask == [1] * first + [0] * 5 + [1] * (len(completion) - first - 5)
    assert completion[first:first + 5] == list('<ENV>')
    assert final == 'so \\boxed{1}'


def test_agent_turns_respects_budget_and_tool_limit():
    def generate(ids, room):
        return list('<run>\nx\n</run>')[:room]
    tool = lambda code: {'success': True, 'timed_out': False, 'stdout': '', 'stderr': ''}  # noqa: E731
    completion, mask, final, used = agent_turns([], generate, tool, lambda m: ['e'] * 3, ''.join,
                                                max_tools=1, budget=40, stop_ids=set())
    assert used == 1 and len(completion) <= 40 and mask.count(0) == 3
    assert final.startswith('<run>')  # the last turn may not call tools; its text is what gets graded


@pytest.mark.parametrize('patched', [True, False])
def test_trl_accepts_agent_rollouts_with_env_mask(tmp_path, monkeypatch, patched):
    import agent_rl
    from datasets import IterableDataset
    from trl import GRPOConfig
    from gos_gemma import attach_gos
    from train_unified import GoSGRPOTrainer, UnifiedConfig
    tok = tokenizer(); tok.padding_side = 'left'
    model = attach_gos(tiny_base(), width=16, nodes=3, rounds=2)
    config = UnifiedConfig(graph_width=16, graph_nodes=3, graph_rounds=2, logits_chunk_size=2,
        max_seq_length=64, max_prompt_tokens=48, max_completion_tokens=6, overlap=4,
        gradient_accumulation_steps=2, grpo_generations=2, agent_max_tools=1)
    if patched:
        monkeypatch.setattr(agent_rl, 'agent_turns', lambda *a, **k: ([9, 7, 7, 9], [1, 0, 0, 1], 'ok', 1))
    seen = []

    def reward(completions, agent_final=None, **kwargs):
        seen.append(agent_final)
        return [float(index % 2) for index in range(len(completions))]
    data = IterableDataset.from_generator(lambda: iter([{'prompt': 'user: a [EOS] assistant: ', 'answer': 'ok'}] * 4))
    args = GRPOConfig(output_dir=str(tmp_path), max_steps=1, per_device_train_batch_size=1,
        gradient_accumulation_steps=2, generation_batch_size=2, num_generations=2, max_completion_length=6,
        beta=0.0, report_to='none', use_cpu=True, bf16=False, use_vllm=False, save_strategy='no', use_cache=False)
    trainer = GoSGRPOTrainer(model=model, args=args, train_dataset=data, processing_class=tok,
        reward_funcs=reward, rollout_func=agent_rl.agent_rollout, gos_config=config)
    trainer.train()
    assert trainer.state.global_step == 1 and seen
    assert all(item == ['ok', 'ok'] for item in seen) if patched else all(len(item) == 2 for item in seen)
    assert math.isfinite(model.get_output_embeddings().lora_B['default'].weight.abs().sum().item())


def test_grpo_reward_grades_agent_final():
    from verified_rewards import grpo_verified_reward
    spec = {'kind': 'math', 'expected': '1'}
    assert grpo_verified_reward(['Output:\n\\boxed{1}', 'x'], task_spec=[spec, spec],
                                agent_final=['\\boxed{2}', '\\boxed{1}']) == [0.0, 1.0]
    with pytest.raises(ValueError):
        grpo_verified_reward(['a'], task_spec=[spec], agent_final=['a', 'b'])
