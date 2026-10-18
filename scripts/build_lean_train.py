"""Lean Workbook proofs (Goedel-Prover) -> Lean SFT/RFT rows whose proofs check on this Mathlib.

Each row's prompt matches the evaluation prompt format (scripts/build_lean_tasks.py) and its answer is the
dataset proof in one ```lean block. Proofs are re-checked here (default heartbeats, axiom audit) with batched
Lean runs; rows whose statement is a near-duplicate (3-gram Jaccard) of an evaluation statement are dropped.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pyarrow.parquet as pq  # noqa: E402
from build_lean_tasks import PROMPT  # noqa: E402
from lean_verifier import DEFAULT_OPENS, check_proofs, normalize_statement  # noqa: E402
from split_audit import canonical, shingles  # noqa: E402

COMMENT = re.compile(r'/-.*?-/', re.S)


def split_proof(full_proof):
    """(statement ending ':= sorry', opens, tactic proof, informal text) or None."""
    informal = ' '.join(re.findall(r'/-(?!-)\s*(.*?)\s*-/', full_proof.split('theorem', 1)[0], re.S))
    match = re.search(r'^(theorem|lemma)\s.*?:=\s*by\b', full_proof, re.S | re.M)
    if not match:
        return None
    statement, opens = normalize_statement(full_proof[:match.end() - len('by')].rstrip() + ' sorry')
    lines = full_proof[match.end():].strip('\n').splitlines()
    indent = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
    proof = '\n'.join(line[indent:] for line in lines).strip('\n')
    return statement, opens or DEFAULT_OPENS, proof, re.sub(r'\s+', ' ', informal).strip()


def body_shingles(statement):
    head = re.sub(r'^\s*(theorem|lemma)\s+\S+', '', statement.strip())
    return shingles(canonical(head))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--against', nargs='*', default=[], help='Evaluation task JSONL files to screen against')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--limit', type=int, help='Random subset size (seeded) before checking')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2, help='Parallel Lean processes (~7 GB RAM each)')
    args = parser.parse_args()
    stats = Counter()
    evaluation = []
    for path in args.against:
        for line in Path(path).read_text().splitlines():
            spec = json.loads(json.loads(line)['task_spec']) if line.strip() else None
            if spec and spec.get('kind') == 'lean':
                evaluation.append(body_shingles(spec['statement']))
    rows = pq.read_table(args.input).to_pylist()
    random.Random(17).shuffle(rows)
    candidates = []
    for row in rows:
        parsed = split_proof(row['full_proof'])
        if parsed is None:
            stats['dropped_unparsed'] += 1
            continue
        grams = body_shingles(parsed[0])
        if any(len(grams & other) / max(1, len(grams | other)) >= args.threshold for other in evaluation):
            stats['dropped_near_duplicate'] += 1
            continue
        candidates.append((row['problem_id'], *parsed))
        if args.limit and len(candidates) >= args.limit:
            break
    groups = {}
    for item in candidates:
        groups.setdefault(item[2], []).append(item)
    batches = [items[i:i + args.batch] for items in groups.values() for i in range(0, len(items), args.batch)]

    def check(batch):
        # Comments do not change elaboration; stripping them keeps batches free of `/-` interference.
        return batch, check_proofs([item[1] for item in batch], [COMMENT.sub('', item[3]) for item in batch],
                                   batch[0][2], confirm=False)

    output = Path(args.output)
    with output.open('w') as handle, ThreadPoolExecutor(args.workers) as pool:
        for done, (batch, results) in enumerate(pool.map(check, batches), 1):
            for (problem_id, statement, opens, proof, informal), result in zip(batch, results):
                stats['kept' if result['correct'] else 'dropped_' + result['reason'].split(':')[0]] += 1
                if not result['correct']:
                    continue
                spec = {'kind': 'lean', 'statement': statement, 'opens': opens}
                handle.write(json.dumps({
                    'id': f'lean_workbook/{problem_id}', 'source': 'lean_workbook', 'cutoff_status': 'pre_cutoff',
                    'prompt': PROMPT.format(statement=statement, informal=informal or 'see the formal statement.'),
                    'answer': f'```lean\n{proof}\n```', 'task_spec': json.dumps(spec, ensure_ascii=False)},
                    ensure_ascii=False) + '\n')
            handle.flush()
            print(json.dumps({'batches': done, 'of': len(batches), **stats}), flush=True)
    Path(str(output) + '.manifest.json').write_text(json.dumps({'input': args.input, 'stats': stats,
                                                                'settings': vars(args)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
