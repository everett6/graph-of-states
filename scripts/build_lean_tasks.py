"""miniF2F (Lean 4) -> task JSONL with lean task specs, keeping statements that compile on this Mathlib."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pyarrow.parquet as pq  # noqa: E402
from lean_verifier import statement_compiles  # noqa: E402

PROMPT = ('Prove the following theorem in Lean 4 with Mathlib.\n\n```lean\n{statement}\n```\n\n'
          'Informal statement: {informal}\n\nReply with the tactic proof that replaces `sorry`, i.e. only the '
          'tactics after `:= by`, in one ```lean block.')


def opens(header):
    lines = [line for line in header.splitlines() if line.startswith('open ')]
    return '\n'.join(lines) or 'open BigOperators Real Nat Topology'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    rows = pq.read_table(args.input).to_pylist()
    def build(row):
        statement = row['formal_statement'].strip()
        if statement.lstrip().startswith('--'):
            return None  # dataset marks it unformalizable
        header = row.get('header') or ''
        if not statement_compiles(statement, opens(header)):
            return None
        informal = row.get('informal_stmt') or re.sub(r'^/--|-/$', '', (row.get('informal_prefix') or '').strip())
        informal = re.sub(r'\s+', ' ', informal).strip()
        return {'id': f"{args.source}/{row.get('id') or row.get('name')}", 'source': args.source, 'cutoff_status': 'pre_cutoff',
                'prompt': PROMPT.format(statement=statement, informal=informal), 'answer': '',
                'task_spec': json.dumps({'kind': 'lean', 'statement': statement, 'opens': opens(header)})}
    with ThreadPoolExecutor(args.workers) as pool:
        built = list(pool.map(build, rows))
    kept = [row for row in built if row]
    Path(args.output).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in kept))
    print(json.dumps({'input': len(rows), 'kept': len(kept), 'dropped_noncompiling': len(rows) - len(kept)}))


if __name__ == '__main__':
    main()
