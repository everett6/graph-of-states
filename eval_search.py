"""Sampling-based evaluation helpers: pass@k, verifier-free majority vote, and summaries.

Math selection must not use the reference at test time, so the deployable math method is
majority vote over symbolically clustered answers (maj@k). pass@k uses the reference and is an
oracle upper bound. For code, selection by the public tests equals scoring by the same tests,
so code best-of-N is also an upper bound until hidden tests exist.
"""
from math import comb
import json


def pass_at_k(n, c, k):
    """Unbiased estimator 1 - C(n-c, k) / C(n, k) (Chen et al., 2021)."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError('Need 0 <= c <= n and 1 <= k <= n')
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def _same(a, b):
    from symbolic_math import equivalent
    if a == b:
        return True
    try:
        return bool(equivalent(a, b))
    except (ValueError, SyntaxError, TypeError, TimeoutError, ZeroDivisionError, AttributeError):
        return False


def majority_vote(answers):
    """Index of a representative of the largest equivalence class; ties go to the earliest class.

    `answers` are extracted final-answer strings (None for unparseable/missing ones)."""
    classes = []  # [representative index, count]
    for index, answer in enumerate(answers):
        if answer is None or not answer.strip():
            continue
        for item in classes:
            if _same(answer, answers[item[0]]):
                item[1] += 1
                break
        else:
            classes.append([index, 1])
    if not classes:
        return None
    return max(classes, key=lambda item: (item[1], -item[0]))[0]


def extracted_answer(completion):
    from verified_rewards import answer_text
    try:
        return answer_text(completion)
    except ValueError:
        return None


def row_metrics(record, ks):
    """record: {'kind', 'samples': [{'correct', 'answer', 'tokens'}], 'repair': optional {'correct', 'tokens'}}"""
    correct = [bool(sample['correct']) for sample in record['samples']]
    n, c = len(correct), sum(correct)
    out = {f'pass@{k}': pass_at_k(n, c, k) for k in ks if k <= n}
    if record['kind'] == 'math':
        for k in ks:
            if k <= n:
                pick = majority_vote([sample['answer'] for sample in record['samples'][:k]])
                out[f'maj@{k}'] = float(pick is not None and correct[pick])
    if 'repair' in record:
        out['any_or_repair'] = float(any(correct) or record['repair']['correct'])
    out['tokens'] = sum(sample['tokens'] for sample in record['samples']) + record.get('repair', {}).get('tokens', 0)
    return out


def wilson(successes, total, z=1.96):
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / (1 + z * z / total)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize(records, ks, group_keys=('source', 'cutoff_status', 'kind')):
    """Mean of every per-row metric overall and per group. CIs are Wilson on pass@1/maj@k-style rates."""
    rows = [(record, row_metrics(record, ks)) for record in records]

    def block(selected):
        names = sorted({name for _, metrics in selected for name in metrics})
        result = {'rows': len(selected)}
        for name in names:
            values = [metrics[name] for _, metrics in selected if name in metrics]
            mean = sum(values) / len(values)
            result[name] = mean
            if name != 'tokens' and all(value in (0.0, 1.0) for value in values):
                result[name + '_ci95'] = wilson(sum(values), len(values))
        return result

    summary = {'overall': block(rows)}
    for key in group_keys:
        for value in sorted({str(record.get(key)) for record, _ in rows}):
            summary[f'{key}={value}'] = block([item for item in rows if str(item[0].get(key)) == value])
    return summary


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]
