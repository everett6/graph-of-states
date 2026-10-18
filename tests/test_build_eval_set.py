import json

import pytest

from build_eval_set import build
from verified_rewards import verify


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return str(path)


def test_build_filters_screens_and_emits_verifiable_specs(tmp_path):
    math = write_jsonl(tmp_path / 'math.jsonl', [
        {'problem_idx': 1, 'problem': 'Find the sum of all positive divisors of 28 minus 28.', 'answer': 28},
        {'problem_idx': 2, 'problem': 'Compute the value of one half plus one quarter exactly.', 'answer': '\\frac{3}{4}'},
        {'problem_idx': 3, 'problem': 'Give the interval where x squared is below one.', 'answer': '(-1,1]'},
        {'problem_idx': 4, 'problem': 'An easy level one problem about adding two and two.', 'answer': 4, 'level': 1}])
    case = {'input': '[1,2,3]\n2', 'output': '5', 'testtype': 'functional'}
    code = write_jsonl(tmp_path / 'lcb.jsonl', [
        {'question_id': 'a', 'question_content': 'Return sum(nums) minus k.', 'starter_code': 'class Solution:',
         'contest_date': '2025-03-01T00:00:00', 'public_test_cases': json.dumps([case]),
         'metadata': json.dumps({'func_name': 'f'}), 'platform': 'leetcode', 'difficulty': 'easy'},
        {'question_id': 'old', 'question_content': 'Old problem.', 'starter_code': '', 'contest_date': '2024-01-01',
         'public_test_cases': json.dumps([case]), 'metadata': json.dumps({'func_name': 'f'})},
        {'question_id': 'stdin', 'question_content': 'Read n.', 'starter_code': '', 'contest_date': '2025-04-01',
         'public_test_cases': json.dumps([{'input': '1', 'output': '1', 'testtype': 'stdin'}]), 'metadata': '{}'}])
    train = write_jsonl(tmp_path / 'train.jsonl', [
        {'messages': [{'role': 'user', 'content': 'Find the sum of all positive divisors of 28 minus 28.'}]}])
    config = {'sources': [
        {'name': 'm', 'adapter': 'math', 'path': math, 'cutoff_status': 'post_cutoff', 'min_level': 0},
        {'name': 'c', 'adapter': 'livecodebench', 'path': code, 'released_after': '2025-01-31',
         'cutoff_status': 'post_cutoff'}]}
    rows, manifest = build(config, [train], threshold=0.5)
    assert [row['id'] for row in rows] == ['m/2', 'm/4', 'c/a']
    assert manifest['stats']['dropped_uncheckable_m'] == 1
    assert manifest['stats']['dropped_near_duplicate'] == 1
    assert manifest['stats']['skipped_c'] == 2
    assert verify('\\boxed{0.75}', rows[0]['task_spec']).correct
    spec = json.loads(rows[2]['task_spec'])
    namespace = {}
    exec('class Solution:\n    def f(self, nums, k):\n        return sum(nums) - k + 1', namespace)
    exec(spec['tests'], namespace)  # trusted test fixture, mirrors the Docker runner's namespace
    assert rows[2]['meta']['private_tests_used'] == 0


def test_private_tests_decode_safely_and_respect_window(tmp_path):
    import base64
    import pickle
    import zlib
    from build_eval_set import livecodebench_rows, private_cases

    def encode(value):
        return base64.b64encode(zlib.compress(pickle.dumps(value))).decode()
    hidden = [{'input': f'[{i}]\n0', 'output': str(i), 'testtype': 'functional'} for i in range(5)]
    assert private_cases(encode(json.dumps(hidden))) == hidden

    class Evil:
        def __reduce__(self):
            return (print, ('pwned',))
    with pytest.raises(pickle.UnpicklingError):
        private_cases(encode(Evil()))
    case = {'input': '[1]\n0', 'output': '1', 'testtype': 'functional'}
    rows = [{'question_id': q, 'question_content': 'Sum.', 'starter_code': 'class Solution:', 'contest_date': d,
             'public_test_cases': json.dumps([case]), 'private_test_cases': encode(json.dumps(hidden)),
             'metadata': json.dumps({'func_name': 'f'})} for q, d in (('in', '2024-06-01'), ('late', '2025-02-01'))]
    path = write_jsonl(tmp_path / 'lcb.jsonl', rows)
    from collections import Counter
    stats = Counter()
    source = {'name': 't', 'path': path, 'released_before': '2025-01-01', 'private_tests': True,
              'max_test_bytes': len(json.dumps([case])) + 3 * len(json.dumps(hidden[0])) + 3}
    built = list(livecodebench_rows(source, stats))
    assert [row['id'] for row in built] == ['t/in'] and stats['skipped_t'] == 1
    assert built[0]['meta'] == {**built[0]['meta'], 'public_tests': 1, 'private_tests_used': 3}


def test_leak_screen_ignores_shared_templates_but_catches_renamed_twins():
    import sys
    from pathlib import Path
    from build_eval_set import jaccard_flags
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
    from build_lean_tasks import PROMPT
    from build_lean_train import near_duplicate, screen_signature
    rows = [{'prompt': PROMPT.format(statement='theorem a : 54 % 6 = 0 := by', informal='Find 54 mod 6.')}]
    jaccard_flags(rows, [PROMPT.format(statement='theorem b : 0 - 0 = 0 := sorry', informal='Evaluate 0 - 0.')], 0.5)
    assert not rows[0]['contamination']['flagged']
    imo = ('theorem imo_1964_p2 (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c) (h₁ : c < a + b) (h₂ : b < a + c)\n'
           '    (h₃ : a < b + c) :\n    a ^ 2 * (b + c - a) + b ^ 2 * (c + a - b) + c ^ 2 * (a + b - c) ≤ 3 * a * b * c := by')
    twin = ('theorem lean_workbook_plus_17090 (a b c : ℝ) (hx: a > 0 ∧ b > 0 ∧ c > 0) (hab : a + b > c) '
            '(hbc : b + c > a) (hca : a + c > b) : a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c'
            '   := sorry')
    assert near_duplicate(screen_signature(twin), screen_signature(imo), 0.5)
    assert not near_duplicate(screen_signature('theorem x : 0 - 0 = 0 := sorry'),
                              screen_signature('theorem y : 54 % 6 = 0 := sorry'), 0.5)
