import json

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
    assert rows[2]['meta']['private_tests_used'] is False
