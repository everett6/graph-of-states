"""Paired comparison of benchmark variants on the same held-out rows.

Every variant in a benchmark run is scored on identical rows, so differences are tested per row: mean
difference in pass@1 (fraction of correct samples) with a seeded paired bootstrap 95% CI, and an exact
two-sided sign test on rows where the two variants disagree. Reported overall and per cutoff group.

    .venv-training/bin/python scripts/compare_heldout.py runs/pilot-heldout-v1 --baseline frozen-d4

--scaling instead compares each recurrent variant's pass counts against its own fewest-pass evaluation
(same weights, more test-time passes): does accuracy rise with passes, including beyond the trained range?
"""
import re
import argparse
import json
from math import comb
from pathlib import Path
import random


def load(run):
    """{variant: {row id: (pass@1, cutoff_status, source)}} for every heldout_records.jsonl under the run;
    recurrent variants have one per evaluated pass count (named '<variant>/passes-N')."""
    variants = {}
    for path in sorted(Path(run).glob('**/heldout_records.jsonl')):
        rows = {}
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                samples = record['samples']
                rows[record['id']] = (sum(bool(s['correct']) for s in samples) / len(samples),
                                      record.get('cutoff_status'), record.get('source'))
        variants[path.parent.relative_to(run).as_posix()] = rows
    return variants


def sign_test(wins, losses):
    """Exact two-sided binomial test p-value for wins vs losses (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a, b, ids, resamples=10000, seed=17):
    """Mean of b - a over ids, its bootstrap 95% CI, wins/losses (rows where b beats / trails a), p-value."""
    diffs = [b[i][0] - a[i][0] for i in ids]
    if not diffs:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(resamples))
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    return {'rows': len(diffs), 'mean_a': sum(a[i][0] for i in ids) / len(ids),
            'mean_b': sum(b[i][0] for i in ids) / len(ids), 'diff': sum(diffs) / len(diffs),
            'ci95': [means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]],
            'wins': wins, 'losses': losses, 'sign_test_p': sign_test(wins, losses)}


def compare(variants, baseline, resamples=10000):
    if baseline not in variants:
        raise ValueError(f'Baseline {baseline!r} not in {sorted(variants)}')
    base = variants[baseline]
    table = {}
    for name, rows in variants.items():
        if name == baseline:
            continue
        shared = sorted(set(base) & set(rows))
        groups = {'all': shared}
        for status in sorted({base[i][1] for i in shared if base[i][1]}):
            groups[f'cutoff={status}'] = [i for i in shared if base[i][1] == status]
        table[name] = {group: paired(base, rows, ids, resamples) for group, ids in groups.items()}
    return table


def scaling(variants, resamples=10000):
    """{variant: {'passes-N': {group: paired}}} against the variant's fewest passes."""
    families = {}
    for name in variants:
        match = re.fullmatch(r'(.+)/(passes-(\d+)(-step[\d.]+)?)', name)
        if match:
            families.setdefault(match.group(1), []).append((int(match.group(3)), match.group(4) or '', match.group(2)))
    table = {}
    for family, entries in sorted(families.items()):
        entries.sort()
        plain = [entry for entry in entries if not entry[1]]
        if len(entries) > 1 and plain:
            subset = {entry[2]: variants[f'{family}/{entry[2]}'] for entry in entries}
            table[family] = compare(subset, plain[0][2], resamples)
    return table


def show(name, groups):
    for group, result in groups.items():
        if result:
            print(f"{name:40s} {group:22s} n={result['rows']:4d} {result['mean_a']:.3f} -> {result['mean_b']:.3f} "
                  f"diff {result['diff']:+.3f} [{result['ci95'][0]:+.3f}, {result['ci95'][1]:+.3f}] "
                  f"W/L {result['wins']}/{result['losses']} p={result['sign_test_p']:.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--baseline', help='Variant directory name to compare against')
    parser.add_argument('--scaling', action='store_true', help='Pass-count scaling within each recurrent variant')
    parser.add_argument('--resamples', type=int, default=10000)
    args = parser.parse_args()
    if args.scaling == bool(args.baseline):
        parser.error('Give exactly one of --baseline or --scaling')
    if args.scaling:
        table = scaling(load(args.run), args.resamples)
        for family, rows in table.items():
            for name, groups in rows.items():
                show(f'{family}/{name}', groups)
        out = Path(args.run) / 'pass_scaling.json'
    else:
        table = compare(load(args.run), args.baseline, args.resamples)
        for name, groups in table.items():
            show(name, groups)
        out = Path(args.run) / f"paired_vs_{args.baseline.replace('/', '_')}.json"
    out.write_text(json.dumps(table, indent=2) + '\n')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
