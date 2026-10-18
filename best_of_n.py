"""Verifier-guided best-of-N with one test-feedback repair turn (roadmap step 3).

For each held-out row: sample N completions, verify each, and for coding rows with no passing
sample, ask for one repair. Writes one JSON line per row (resumable) and a summary with
pass@k (oracle), maj@k (deployable math selection), repair rate, tokens and seconds.
Reads local weights only.
"""
import argparse
import json
from pathlib import Path
import time

from eval_search import extracted_answer, read_jsonl, summarize
from verified_rewards import verify, verify_many

REPAIR_PROMPT = ('Your solution failed the provided unit tests.\nVerifier feedback: {feedback}\n'
                 'Find the bug and return the complete corrected solution in one ```python block.')


def evaluate_row(row, generate, n, repair=True, verifier=verify):
    """generate(messages, count, sample) -> list of (text, token_count)."""
    messages = [{'role': 'user', 'content': row['prompt']}]
    spec = json.loads(row['task_spec']) if isinstance(row['task_spec'], str) else row['task_spec']
    start = time.perf_counter()
    samples = []
    outputs = generate(messages, n, True)
    texts = [text for text, _ in outputs]
    verdicts = verify_many(texts, [spec] * len(texts)) if verifier is verify else [verifier(text, spec) for text in texts]
    for (text, tokens), verdict in zip(outputs, verdicts):
        if not verdict.verified:
            raise RuntimeError(f"Verifier unavailable for {row['id']}: {verdict.reason}")
        samples.append({'text': text, 'tokens': tokens, 'correct': verdict.correct, 'reason': verdict.reason,
                        'answer': extracted_answer(text) if spec['kind'] == 'math' else None})
    record = {key: row.get(key) for key in ('id', 'source', 'cutoff_status')}
    record.update(kind=spec['kind'], samples=samples)
    if repair and spec['kind'] == 'coding' and not any(sample['correct'] for sample in samples):
        failed = samples[0]
        prompt = REPAIR_PROMPT.format(feedback=failed['reason'].split(': ', 1)[-1][:1000])
        (text, tokens), = generate(messages + [{'role': 'assistant', 'content': failed['text']},
                                               {'role': 'user', 'content': prompt}], 1, False)
        verdict = verifier(text, spec)
        record['repair'] = {'text': text, 'tokens': tokens, 'correct': verdict.correct, 'reason': verdict.reason}
    record['seconds'] = time.perf_counter() - start
    return record


def run(rows, generate, n, ks, output, repair=True, verifier=verify):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / 'records.jsonl'
    done = {record['id'] for record in read_jsonl(records_path)} if records_path.exists() else set()
    with records_path.open('a') as handle:
        for row in rows:
            if row['id'] in done:
                continue
            record = evaluate_row(row, generate, n, repair, verifier)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            print(json.dumps({'id': record['id'], 'correct': sum(s['correct'] for s in record['samples']),
                              'n': n, 'seconds': round(record['seconds'], 1)}), flush=True)
    records = read_jsonl(records_path)
    summary = summarize(records, ks)
    summary['seconds_total'] = sum(record['seconds'] for record in records)
    return summary


def hf_generator(model, tokenizer, max_new_tokens, temperature, batch, thinking=False, context=4096):
    import torch

    @torch.no_grad()
    def generate(messages, count, sample):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=thinking)
        ids = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
        budget = min(max_new_tokens, context - ids.shape[1])
        if budget < 1:
            raise ValueError('Prompt exceeds the context budget')
        results = []
        for offset in range(0, count, batch):
            size = min(batch, count - offset)
            kwargs = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0) if sample else dict(do_sample=False)
            output = model.generate(input_ids=ids.repeat(size, 1), attention_mask=torch.ones_like(ids).repeat(size, 1),
                                    max_new_tokens=budget, pad_token_id=tokenizer.pad_token_id, use_cache=True, **kwargs)
            for sequence in output[:, ids.shape[1]:]:
                tokens = sequence.tolist()
                length = next((i + 1 for i, t in enumerate(tokens) if t == tokenizer.eos_token_id), len(tokens))
                results.append((tokenizer.decode(tokens[:length], skip_special_tokens=True), length))
        return results
    return generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', default='data/eval/heldout-v1.jsonl')
    parser.add_argument('--base-path', required=True, help='Local Gemma snapshot directory')
    parser.add_argument('--adapter', help='Optional adapter directory (GoS save_gos_adapter or PEFT LoRA)')
    parser.add_argument('--adapter-kind', choices=['gos', 'lora', 'recurrent'], default='gos')
    parser.add_argument('--passes', type=int, help='Recurrent pass count at inference')
    parser.add_argument('--output', required=True)
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--ks', default='1,2,4,8')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--no-repair', action='store_true')
    parser.add_argument('--sources', help='Comma-separated source names to keep')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    ks = [int(k) for k in args.ks.split(',')]
    if args.n < max(ks) or args.n < 1 or args.temperature <= 0:
        raise ValueError('--n must be >= max(--ks) and temperature positive')
    import torch
    from local_gemma import load_causal_lm
    torch.manual_seed(args.seed)
    model, tokenizer = load_causal_lm(args.base_path)
    if args.adapter and args.adapter_kind == 'recurrent':
        from gos_recurrent import load_recurrence
        load_recurrence(model, args.adapter, args.passes)  # before PEFT so module names match
    if args.adapter and args.adapter_kind == 'gos':
        from gos_gemma import load_gos_adapter
        model = load_gos_adapter(model, args.adapter).eval()
    elif args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    rows = read_jsonl(args.eval)
    if args.sources:
        keep = set(args.sources.split(','))
        rows = [row for row in rows if row['source'] in keep]
    rows = rows[:args.limit] if args.limit else rows
    generate = hf_generator(model, tokenizer, args.max_new_tokens, args.temperature, args.batch, args.thinking)
    summary = run(rows, generate, args.n, ks, args.output, not args.no_repair)
    summary['settings'] = vars(args)
    (Path(args.output) / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary['overall'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
