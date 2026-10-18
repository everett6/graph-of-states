"""Frozen-backbone headroom on the held-out eval set (no adapter, greedy, no training).

Generates one greedy completion per row (resumable, appended to completions.jsonl), then
scores every completion with verified_rewards.verify and writes pass@1 per source and per
cutoff_status with 95% Wilson intervals to summary.json. `--score-only` rescores saved
completions (e.g. once the Docker verifier image is available for coding rows).
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verified_rewards import verify  # noqa: E402

BASE = ('/home/everett/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-q4_0-unquantized/'
        'snapshots/b6ed86275a6a5735884e208bfed95b445a684ca2')


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def generate(rows, out, args):
    import torch
    from local_gemma import load_local_gemma
    done = {row['id'] for row in load_jsonl(out)}
    todo = [row for row in rows if row['id'] not in done]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        return
    model, tokenizer = load_local_gemma(args.base)
    tokenizer.padding_side = 'left'
    rendered = {row['id']: tokenizer.apply_chat_template(
        [{'role': 'user', 'content': row['prompt']}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) for row in todo}
    lengths = {key: len(tokenizer.encode(text, add_special_tokens=False))
               for key, text in rendered.items()}
    # Similar lengths share a batch (less padding); longest first so any OOM shows up at once.
    todo.sort(key=lambda row: -lengths[row['id']])
    configured = model.generation_config.eos_token_id
    eos = {tokenizer.eos_token_id, tokenizer.pad_token_id,
           *(configured if isinstance(configured, list) else [configured])} - {None}
    with Path(out).open('a') as stream:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            inputs = tokenizer([rendered[row['id']] for row in batch], return_tensors='pt',
                               padding=True, add_special_tokens=False).to(model.device)
            torch.cuda.synchronize()
            began = time.perf_counter()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                        eos_token_id=sorted(eos - {tokenizer.pad_token_id}),
                                        pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            for row, sequence in zip(batch, output[:, inputs.input_ids.shape[1]:]):
                ids = sequence.tolist()
                stop = next((i for i, token in enumerate(ids) if token in eos), None)
                kept = ids if stop is None else ids[:stop]
                stream.write(json.dumps({
                    'id': row['id'], 'source': row['source'], 'cutoff_status': row['cutoff_status'],
                    'completion': tokenizer.decode(kept, skip_special_tokens=True),
                    'tokens': len(kept), 'truncated': stop is None,
                    'prompt_tokens': lengths[row['id']], 'batch_size': len(batch),
                    'batch_seconds': elapsed, 'seconds_per_row': elapsed / len(batch)}) + '\n')
            stream.flush()
            print(f'{start + len(batch)}/{len(todo)} rows, batch {elapsed:.0f}s, '
                  f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB', flush=True)


def score(rows, out, summary_path, args):
    specs = {row['id']: row['task_spec'] for row in rows}
    results = []
    for record in load_jsonl(out):
        verdict = verify(record['completion'], specs[record['id']])
        results.append({**{k: record[k] for k in ('id', 'source', 'cutoff_status', 'tokens',
                                                   'truncated', 'seconds_per_row')},
                        **verdict.as_dict()})
    Path(summary_path).with_name('verdicts.jsonl').write_text(
        ''.join(json.dumps(result) + '\n' for result in results))

    def group(items):
        scored = [item for item in items if item['verified']]
        k, n = sum(item['correct'] for item in scored), len(scored)
        return {'pass': k, 'n': n, 'unverified': len(items) - n, 'rate': k / n if n else None,
                'wilson95': wilson(k, n), 'truncated': sum(item['truncated'] for item in items),
                'mean_tokens': sum(item['tokens'] for item in items) / max(1, len(items))}

    summary = {'n_rows_total': len(rows), 'n_generated': len(results),
               'max_new_tokens': args.max_new_tokens, 'batch_size': args.batch_size,
               'decoding': 'greedy, enable_thinking=False, NF4 frozen backbone, no adapter',
               'overall': group(results),
               'post_cutoff_math': group([r for r in results if r['cutoff_status'] == 'post_cutoff'
                                          and r['source'] != 'lcb_v6_leetcode']),
               'by_source': {s: group([r for r in results if r['source'] == s])
                             for s in sorted({r['source'] for r in results})},
               'by_cutoff_status': {c: group([r for r in results if r['cutoff_status'] == c])
                                    for c in sorted({r['cutoff_status'] for r in results})},
               'seconds_per_row_mean': sum(r['seconds_per_row'] for r in results) / max(1, len(results))}
    Path(summary_path).write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/eval/heldout-v1.jsonl')
    parser.add_argument('--out-dir', default='runs/headroom-v1')
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='Generate at most N new rows')
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    rows = load_jsonl(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'completions.jsonl'
    if not args.score_only:
        generate(rows, out, args)
    score(rows, out, out_dir / 'summary.json', args)


if __name__ == '__main__':
    main()
