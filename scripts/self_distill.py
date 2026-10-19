"""Self-distilled SFT rows: the frozen model's own verified solutions (rejection-sampling fine-tuning).

SFT on MATH-train's human-written solutions made the pilot's LoRA arm imitate their terse style and drop from
52.8% to 15.2% held-out pass@1 (runs/pilot-heldout-v1, paired, n=125). Training on solutions the model itself
wrote and the verifier accepted keeps its reasoning style (RFT / STaR). For each sampled task row: generate n
completions with the same chat rendering the benchmark trains on (thinking off), keep the shortest verified
one that fits the token budget. Resumable; the row order is a seeded shuffle, never chosen by outcomes.

    .venv-training/bin/python scripts/self_distill.py --input data/train/math-train-v1.jsonl \
        --output data/train/math-selfdistill-1024.jsonl --base-path <gemma dir> --sample 700 --budget 1024
"""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_search import read_jsonl  # noqa: E402
from verified_rewards import verify_many  # noqa: E402


def row_id(row):
    return row.get('trace_id') or row.get('id')


def distill(rows, generate, n, fits, done=()):
    """Yield (row id, output row or None). generate(messages, count, sample) -> [(text, tokens)];
    fits(prompt, answer) -> bool. Rows whose id is in `done` are skipped."""
    for row in rows:
        if row_id(row) in done:
            continue
        spec = json.loads(row['task_spec']) if isinstance(row['task_spec'], str) else row['task_spec']
        outputs = generate([{'role': 'user', 'content': row['prompt']}], n, True)
        verdicts = verify_many([text for text, _ in outputs], [spec] * len(outputs))
        good = sorted((tokens, text) for (text, tokens), verdict in zip(outputs, verdicts)
                      if verdict.verified and verdict.correct and fits(row['prompt'], text))
        if not good:
            yield row_id(row), None
            continue
        yield row_id(row), {'prompt': row['prompt'], 'answer': good[0][1], 'task_spec': spec,
                            'source': f"selfdistill_{row.get('source', 'task')}", 'trace_id': row_id(row),
                            'correct_samples': sum(v.correct for v in verdicts), 'samples': len(outputs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--base-path', required=True)
    parser.add_argument('--sample', type=int, help='Seeded random subset of input rows')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--budget', type=int, required=True, help='Rendered prompt + answer + EOS token limit')
    # Gemma 4's recommended sampling, so the kept solutions come from the distribution the model was tuned for.
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=64)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    import torch
    from best_of_n import hf_generator
    from local_gemma import load_causal_lm
    torch.manual_seed(args.seed)
    model, tokenizer = load_causal_lm(args.base_path)
    model.eval()
    rows = read_jsonl(args.input)
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.sample] if args.sample else rows

    def prefix_length(prompt):
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    def fits(prompt, answer):
        return prefix_length(prompt) + len(tokenizer.encode(answer, add_special_tokens=False)) + 1 <= args.budget

    output = Path(args.output)
    log = Path(str(output) + '.attempts.jsonl')  # every attempted id, so resume skips failures too
    done = {json.loads(line)['id'] for line in log.read_text().splitlines() if line.strip()} if log.exists() else set()
    longest = max(1, args.budget - min((prefix_length(row['prompt']) for row in rows[:50]), default=0))
    generate = hf_generator(model, tokenizer, longest, args.temperature, args.n, False, args.budget + 64,
                            args.top_k, args.top_p)
    kept = attempted = 0
    with output.open('a') as handle, log.open('a') as attempts:
        for identifier, result in distill(rows, generate, args.n, fits, done):
            attempted += 1
            if result:
                kept += 1
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                handle.flush()
            attempts.write(json.dumps({'id': identifier, 'kept': bool(result)}) + '\n')
            attempts.flush()
            if attempted % 25 == 0:
                print(json.dumps({'attempted': attempted, 'kept': kept, 'resumed_skip': len(done)}), flush=True)
    print(json.dumps({'attempted': attempted, 'kept': kept, 'resumed_skip': len(done)}), flush=True)


if __name__ == '__main__':
    main()
