"""Build a held-out math/code eval set with verifier task specs; reads local files only.

Each output row has `prompt`, `answer`, `source`, `group`, `contamination` and a JSON-string
`task_spec` accepted by `verified_rewards.verify` / `grpo_verified_reward`.
LiveCodeBench private tests are pickled and are never loaded; only public JSON tests are used.
"""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path

from split_audit import canonical, shingles
from symbolic_math import Unsupported, parse_answer

MATH_SUFFIX = '\n\nReason step by step, then put only the final answer in \\boxed{}.'
CODE_SUFFIX = ('\n\nWrite a complete Python 3 solution in one ```python block. Include every import you '
               'use (for example `from typing import List`). Do not read input or print output.')


def read_rows(path):
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def math_rows(source, stats):
    for row in read_rows(source['path']):
        if 'min_level' in source and int(row.get('level', 0)) < source['min_level']:
            continue
        answer = str(row['answer']).strip()
        try:
            parse_answer(answer)
        except (Unsupported, SyntaxError, TypeError, ValueError, ZeroDivisionError):
            stats[f"dropped_uncheckable_{source['name']}"] += 1  # e.g. intervals, text answers
            continue
        ident = row.get('unique_id', row.get('problem_idx'))
        yield {'id': f"{source['name']}/{ident}", 'prompt': row['problem'].strip() + MATH_SUFFIX,
               'answer': answer, 'task_spec': {'kind': 'math', 'expected': answer},
               'meta': {key: row[key] for key in ('level', 'subject') if key in row}}


def leetcode_tests(func_name, cases):
    """Assertion script for docker_code_reward: args are newline-separated JSON values."""
    pairs = [[case['input'], case['output']] for case in cases]
    return ('import json\n'
            f'_cases = json.loads({json.dumps(json.dumps(pairs))})\n'
            'for _args_text, _expected_text in _cases:\n'
            "    _args = [json.loads(line) for line in _args_text.split('\\n') if line.strip()]\n"
            f'    assert Solution().{func_name}(*_args) == json.loads(_expected_text)\n')


def livecodebench_rows(source, stats):
    after = date.fromisoformat(source['released_after'])
    for row in read_rows(source['path']):
        released = date.fromisoformat(str(row['contest_date'])[:10])
        cases = json.loads(row['public_test_cases'])
        if released <= after or not cases or any(case.get('testtype') != 'functional' for case in cases):
            stats[f"skipped_{source['name']}"] += 1  # pre-cutoff, or stdin/stdout needing another harness
            continue
        func = json.loads(row.get('metadata') or '{}').get('func_name')
        if not func or not func.isidentifier():
            raise ValueError('Functional problem without a valid func_name')
        prompt = (row['question_content'].strip() + '\n\nStarter code:\n```python\n' +
                  row['starter_code'].strip() + '\n```' + CODE_SUFFIX)
        tests = leetcode_tests(func, cases)
        yield {'id': f"{source['name']}/{row['question_id']}", 'prompt': prompt, 'answer': '',
               'tests': tests, 'task_spec': {'kind': 'coding', 'verifier': 'docker', 'tests': tests},
               'meta': {'contest_date': str(released), 'difficulty': row.get('difficulty'),
                        'platform': row.get('platform'), 'public_tests': len(cases), 'private_tests_used': False}}


ADAPTERS = {'math': math_rows, 'livecodebench': livecodebench_rows}


def reference_prompts(paths):
    for path in paths:
        for row in read_rows(path):
            if 'prompt' in row:
                yield row['prompt']
            else:
                messages = row.get('messages', row.get('conversations', []))
                yield ' '.join(str(m.get('content', m.get('value', ''))) for m in messages
                               if m.get('role', m.get('from')) in ('user', 'human'))


def jaccard_flags(rows, against, threshold):
    """Max 3-gram Jaccard of each row's problem text vs. reference prompts and earlier rows."""
    inverted, signatures = {}, []
    for text in against:
        signature = shingles(text)
        signatures.append(signature)
        for item in signature:
            inverted.setdefault(item, []).append(len(signatures) - 1)
    external = len(signatures)
    for row in rows:
        problem = row['prompt'].split(MATH_SUFFIX)[0].split('\n\nStarter code:')[0]
        signature = shingles(problem)
        best, internal = 0.0, False
        for other in {index for item in signature for index in inverted.get(item, ())}:
            score = len(signature & signatures[other]) / len(signature | signatures[other])
            if score > best:
                best, internal = score, other >= external
        row['contamination'] = {'max_jaccard': round(best, 4), 'flagged': best >= threshold,
                                'match': 'eval_duplicate' if internal else 'reference'}
        signatures.append(signature)
        for item in signature:
            inverted.setdefault(item, []).append(len(signatures) - 1)


def build(config, against=(), threshold=0.5):
    rows, stats = [], Counter()
    for source in config['sources']:
        for row in ADAPTERS[source['adapter']](source, stats):
            row.update(source=source['name'], cutoff_status=source['cutoff_status'],
                       group=hashlib.sha256(canonical(row['prompt']).encode()).hexdigest()[:16])
            rows.append(row)
            stats[f"read_{source['name']}"] += 1
    jaccard_flags(rows, list(reference_prompts(against)), threshold)
    kept = [row for row in rows if not row['contamination']['flagged']]
    stats['dropped_near_duplicate'] = len(rows) - len(kept)
    for row in kept:
        row['task_spec'] = json.dumps(row['task_spec'], sort_keys=True)
    payload = json.dumps(kept, sort_keys=True, separators=(',', ':'))
    manifest = {'rows': len(kept), 'stats': dict(stats), 'jaccard_threshold': threshold,
                'reference_files': [str(path) for path in against],
                'by_source': dict(Counter(row['source'] for row in kept)),
                'by_cutoff_status': dict(Counter(row['cutoff_status'] for row in kept)),
                'sha256': hashlib.sha256(payload.encode()).hexdigest(),
                'headroom': 'not measured; run the frozen backbone on this file before using it'}
    return kept, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True, help='Output JSONL; a .manifest.json is written beside it')
    parser.add_argument('--against', nargs='*', default=[], help='Training JSONL/parquet files to screen against')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    rows, manifest = build(config, args.against, args.threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows))
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
