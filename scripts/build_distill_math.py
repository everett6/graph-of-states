"""OpenR1-Math-220k (DeepSeek-R1 traces, Apache 2.0) -> SFT rows in Gemma 4's thinking format.

Each kept row is one complete trace that Math-Verify marked correct and that our own verifier accepts against
the dataset answer. Its `<think>...</think>` reasoning goes into Gemma's thought channel and the summary
after it becomes the final answer:

    prompt (rendered with thinking on) -> <|channel>thought\\n{reasoning}<channel|>{final}<turn|>

Rows must fit --budget tokens (prompt + answer + end token). Rows whose problem is a near-duplicate
(3-gram Jaccard >= --threshold) of an evaluation prompt or of an earlier kept row are dropped. Only
single-answer word problems are kept (no proofs or multiple choice).

    .venv-training/bin/python scripts/build_distill_math.py --input data/distill/openr1/data \\
        --output data/train/math-openr1-3072.jsonl --base-path <gemma dir> --budget 3072 \\
        --against data/eval/heldout-v2.jsonl data/eval/heldout-v1.jsonl data/lean/minif2f-test.jsonl
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_eval_set import MATH_SUFFIX, jaccard_flags  # noqa: E402
from verified_rewards import verify_many  # noqa: E402

THINK = re.compile(r'^\s*<think>\n?(.*?)\n?</think>\s*(.*)$', re.S)


def to_gemma(generation):
    """'<think>R</think> F' -> '<|channel>thought\\nR<channel|>F', or None if malformed or empty."""
    match = THINK.match(generation)
    if not match or not match.group(1).strip() or not match.group(2).strip() or '<think>' in match.group(2):
        return None
    return f'<|channel>thought\n{match.group(1).strip()}<channel|>{match.group(2).strip()}'


def candidates(rows):
    """(row, gemma answer) for each verified-complete trace of a single-answer word problem; the shortest
    such trace per problem first."""
    for row in rows:
        if row.get('question_type') != 'math-word-problem' or not (row.get('answer') or '').strip():
            continue
        flags = zip(row['generations'], row.get('correctness_math_verify') or [],
                    row.get('is_reasoning_complete') or [])
        traces = sorted((text for text, correct, complete in flags if correct and complete), key=len)
        converted = [answer for answer in map(to_gemma, traces) if answer]
        if converted:
            yield row, converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Directory of OpenR1-Math parquet shards')
    parser.add_argument('--output', required=True)
    parser.add_argument('--base-path', required=True, help='Model directory with the tokenizer')
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--against', nargs='*', default=[], help='Evaluation JSONL files to screen against')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--limit', type=int, help='Stop after this many kept rows')
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_path)
    rows = [row for path in sorted(Path(args.input).glob('*.parquet')) for row in pq.read_table(path).to_pylist()]
    random.Random(args.seed).shuffle(rows)
    stats = Counter(read=len(rows))
    against = [json.loads(line)['prompt'] for path in args.against for line in Path(path).read_text().splitlines()
               if line.strip()]
    kept = []
    for row, answers in candidates(rows):
        prompt = row['problem'].strip() + MATH_SUFFIX
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=True)
        room = args.budget - len(tokenizer.encode(rendered, add_special_tokens=False)) - 1
        answer = next((a for a in answers if len(tokenizer.encode(a, add_special_tokens=False)) <= room), None)
        if answer is None:
            stats['dropped_over_budget'] += 1
            continue
        kept.append({'prompt': prompt, 'answer': answer, 'thinking': True,
                     'task_spec': {'kind': 'math', 'expected': row['answer'].strip()},
                     'source': f"openr1_{row.get('source', 'math')}", 'trace_id': f"openr1/{row['uuid']}"})
    stats['fit_budget'] = len(kept)
    jaccard_flags(kept, against, args.threshold)
    screened = [row for row in kept if not row.pop('contamination')['flagged']]
    stats['dropped_near_duplicate'] = len(kept) - len(screened)
    verdicts = verify_many([row['answer'] for row in screened], [row['task_spec'] for row in screened])
    final = [row for row, verdict in zip(screened, verdicts) if verdict.verified and verdict.correct]
    stats['dropped_our_verifier'] = len(screened) - len(final)
    final = final[:args.limit] if args.limit else final
    stats['kept'] = len(final)
    Path(args.output).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in final))
    Path(args.output + '.manifest.json').write_text(json.dumps(
        {'source': 'open-r1/OpenR1-Math-220k (default, shards in --input)', 'license': 'apache-2.0',
         'stats': stats, 'settings': vars(args)}, indent=2) + '\n')
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
