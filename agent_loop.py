"""Multi-turn tool use: the model runs Python in a sandbox, reads the output, and continues.

Protocol (plain text, no JSON escaping):  <run>\n...python...\n</run>  -> generation stops at </run>,
the code runs in the sandbox (Docker or bubblewrap, no network), and the bounded output comes back as
the next user turn. A turn without <run> is the final answer, checked by the task verifier.
No public or hidden test cases are exposed to the agent, so agentic scores stay comparable to
single-shot scores. Also collects verified successful trajectories for rejection-sampling SFT.
"""
import argparse
import json
from pathlib import Path
import re
import time

from eval_search import extracted_answer, read_jsonl, summarize
from verified_rewards import verify

AGENT_INSTRUCTIONS = (
    '\n\nYou can run Python 3 (standard library only, no internet) to calculate or to test your code. '
    'To run code, write it between <run> and </run> on their own lines, then stop and wait: '
    'the output will be shown to you. Use at most {tools} runs. When you are confident, reply without '
    '<run> and give the final answer in the requested format.')
LEAN_INSTRUCTIONS = (
    '\n\nYou can check a proof attempt with Lean: write the tactic proof between <lean> and </lean> on their own '
    'lines, then stop and wait; Lean\'s response (errors and remaining goals) will be shown to you. Use at most '
    '{tools} checks. Then reply without <lean> and give the final tactic proof in one ```lean block.')
RUN = re.compile(r'<run>\s*\n?(.*?)</run>', re.S)
LEAN = re.compile(r'<lean>\s*\n?(.*?)</lean>', re.S)
OUTPUT_LIMIT = 2000


def tool_call(text, pattern=RUN):
    """(code, text truncated after the call) or (None, text)."""
    match = pattern.search(text)
    if not match or not match.group(1).strip():
        return None, text
    return match.group(1), text[:match.end()]


def lean_executor(spec):
    """Tool backend for Lean tasks: check an attempted tactic proof of the task's own statement."""
    from lean_verifier import DEFAULT_OPENS, check_proof, feedback
    def execute(proof, timeout=120, output_limit=OUTPUT_LIMIT):
        result = check_proof(spec['statement'], proof, spec.get('opens', DEFAULT_OPENS), timeout)
        return {'success': result['correct'], 'timed_out': result['reason'] == 'timeout',
                'stdout': feedback(result, output_limit), 'stderr': ''}
    return execute


def run_agent(prompt, generate, max_tools=3, executor=None, kind='python'):
    """generate(messages, sample) -> (text, tokens). Returns the final text and the full transcript.
    kind='python' runs <run> code; kind='lean' checks <lean> proofs (executor from lean_executor)."""
    if executor is None:
        from gos_tools import execute_python as executor
    pattern, instructions = (LEAN, LEAN_INSTRUCTIONS) if kind == 'lean' else (RUN, AGENT_INSTRUCTIONS)
    messages = [{'role': 'user', 'content': prompt + instructions.format(tools=max_tools)}]
    tokens, calls = 0, []
    for turn in range(max_tools + 1):
        text, used = generate(messages, True)
        tokens += used
        code, text = tool_call(text, pattern) if turn < max_tools else (None, text)
        messages.append({'role': 'assistant', 'content': text})
        if code is None:
            return {'final': text, 'messages': messages, 'tokens': tokens, 'tool_calls': calls}
        outcome = executor(code, timeout=120 if kind == 'lean' else 10, output_limit=OUTPUT_LIMIT)
        feedback = (outcome['stdout'] + ('\n' + outcome['stderr'] if outcome['stderr'].strip() else '')).strip()
        if outcome['timed_out']:
            feedback = 'TimeoutError: the check ran out of time'
        calls.append({'success': outcome['success'], 'timed_out': outcome['timed_out']})
        remaining = max_tools - turn - 1
        messages.append({'role': 'user', 'content': f'Output:\n{feedback[:OUTPUT_LIMIT] or "(no output)"}\n\n' + (
            f'You have {remaining} run(s) left.' if remaining else 'No runs left: give your final answer now.')})
    return {'final': messages[-1]['content'], 'messages': messages, 'tokens': tokens, 'tool_calls': calls}


def evaluate_agent_row(row, generate, n, max_tools=3, executor=None, verifier=verify):
    spec = json.loads(row['task_spec']) if isinstance(row['task_spec'], str) else row['task_spec']
    start = time.perf_counter()
    samples = []
    for _ in range(n):
        kind = 'lean' if spec['kind'] == 'lean' else 'python'
        tool = lean_executor(spec) if kind == 'lean' and executor is None else executor
        result = run_agent(row['prompt'], generate, max_tools, tool, kind)
        verdict = verifier(result['final'], spec)
        if not verdict.verified:
            raise RuntimeError(f"Verifier unavailable for {row['id']}: {verdict.reason}")
        samples.append({'text': result['final'], 'messages': result['messages'], 'tokens': result['tokens'],
                        'tool_calls': len(result['tool_calls']), 'correct': verdict.correct,
                        'reason': verdict.reason[:500],
                        'answer': extracted_answer(result['final']) if spec['kind'] == 'math' else None})
    record = {key: row.get(key) for key in ('id', 'source', 'cutoff_status')}
    record.update(kind=spec['kind'], samples=samples, seconds=time.perf_counter() - start,
                  mean_tool_calls=sum(s['tool_calls'] for s in samples) / n)
    return record


def trajectories(records, rows):
    """Verified-correct agent transcripts as SFT rows (assistant turns supervised, tool output not)."""
    by_id = {row['id']: row for row in rows}
    for record in records:
        for index, sample in enumerate(record['samples']):
            if sample['correct']:
                row = by_id[record['id']]
                yield {'messages': sample['messages'], 'task_spec': row['task_spec'], 'source': 'agent_rft',
                       'trace_id': f"{record['id']}#{index}"}


def chat_generator(model, tokenizer, max_new_tokens, temperature, thinking=False, context=4096):
    import torch

    @torch.no_grad()
    def generate(messages, sample):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=thinking)
        ids = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
        budget = min(max_new_tokens, context - ids.shape[1])
        if budget < 1:
            return 'Context budget exhausted.', 0
        kwargs = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0) if sample else dict(do_sample=False)
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=budget,
                                pad_token_id=tokenizer.pad_token_id, use_cache=True, stop_strings=['</run>', '</lean>'],
                                tokenizer=tokenizer, **kwargs)
        tokens = output[0, ids.shape[1]:].tolist()
        length = next((i + 1 for i, t in enumerate(tokens) if t == tokenizer.eos_token_id), len(tokens))
        return tokenizer.decode(tokens[:length], skip_special_tokens=True), length
    return generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', required=True, help='Task JSONL (held-out set, or training tasks with --collect)')
    parser.add_argument('--base-path', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--adapter-kind', choices=['gos', 'lora', 'recurrent'], default='lora')
    parser.add_argument('--passes', type=int, help='Recurrent pass count at inference')
    parser.add_argument('--output', required=True)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--max-tools', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max-new-tokens', type=int, default=1536)
    parser.add_argument('--context', type=int, default=6144)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--sources')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sample', type=int, help='Seeded random subset of this many rows (after --sources)')
    parser.add_argument('--collect', help='Write verified successful transcripts here (training tasks only)')
    args = parser.parse_args()
    import torch
    from local_gemma import load_causal_lm
    torch.manual_seed(17)
    model, tokenizer = load_causal_lm(args.base_path)
    if args.adapter_kind == 'recurrent' and args.adapter:
        from gos_recurrent import load_recurrence
        load_recurrence(model, args.adapter, args.passes)
    if args.adapter and args.adapter_kind == 'gos':
        from gos_gemma import load_gos_adapter
        model = load_gos_adapter(model, args.adapter).eval()
    elif args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    rows = read_jsonl(args.eval)
    if args.sources:
        rows = [row for row in rows if row['source'] in set(args.sources.split(','))]
    if args.sample:
        import random
        rows = sorted(random.Random(17).sample(rows, min(args.sample, len(rows))), key=lambda row: row['id'])
    rows = rows[:args.limit] if args.limit else rows
    generate = chat_generator(model, tokenizer, args.max_new_tokens, args.temperature, args.thinking, args.context)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'records.jsonl'
    done = {record['id'] for record in read_jsonl(path)} if path.exists() else set()
    with path.open('a') as handle:
        for row in rows:
            if row['id'] in done:
                continue
            record = evaluate_agent_row(row, generate, args.n, args.max_tools)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            print(json.dumps({'id': row['id'], 'correct': sum(s['correct'] for s in record['samples']),
                              'tools': record['mean_tool_calls'], 'seconds': round(record['seconds'])}), flush=True)
    records = read_jsonl(path)
    ks = [k for k in (1, 2, 4, 8) if k <= args.n]
    summary = summarize(records, ks)
    summary['mean_tool_calls'] = sum(r['mean_tool_calls'] for r in records) / len(records)
    summary['settings'] = vars(args)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    if args.collect:
        Path(args.collect).write_text(''.join(json.dumps(item, ensure_ascii=False) + '\n'
                                              for item in trajectories(records, rows)))
    print(json.dumps(summary['overall'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
