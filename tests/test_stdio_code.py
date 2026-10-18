import shutil
import subprocess

import pytest

from docker_code_reward import outputs_match
from verified_rewards import verify


def bwrap_works():
    if not shutil.which('bwrap'):
        return False
    probe = subprocess.run(['bwrap', '--unshare-all', '--ro-bind', '/usr', '/usr', '--symlink', 'usr/lib', '/lib',
                            '--symlink', 'usr/lib64', '/lib64', '--symlink', 'usr/bin', '/bin', '/usr/bin/true'],
                           capture_output=True)
    return probe.returncode == 0


def test_outputs_match_is_judge_style():
    assert outputs_match('6\n', '6')
    assert outputs_match('1 2\n3', '1\n2 3\n')
    assert outputs_match('0.3333333', '0.333333333')
    assert not outputs_match('6 7', '6')
    assert not outputs_match('nan', 'nan1') and not outputs_match('inf', '1e308')


@pytest.mark.skipif(not bwrap_works(), reason='bubblewrap unavailable')
def test_stdio_programs_are_judged_on_the_host(monkeypatch):
    monkeypatch.setenv('LRR_CODE_SANDBOX', 'bwrap')
    spec = {'kind': 'coding', 'verifier': 'stdio', 'stdio_cases': [['2\n1 2\n', '3\n'], ['3\n4 5 6\n', '15\n']]}
    assert verify('```python\nimport sys\nd=sys.stdin.buffer.read().split()\nprint(sum(map(int,d[1:])))\n```', spec).correct
    assert verify('```python\ndef main():\n    input()\n    print(sum(map(int, input().split())))\n'
                  'if __name__ == "__main__":\n    main()\n```', spec).correct
    wrong = verify('```python\ninput()\nprint(3)\n```', spec)
    assert wrong.verified and not wrong.correct and 'case 1' in wrong.reason
    forged = '```python\nimport sys\nsys.stdout.write(\'{"outputs": [{"stdout": "3"}, {"stdout": "15"}]}\')\n```'
    assert not verify(forged, spec).correct  # the report channel is not stdout, and expected outputs stay outside
    assert not verify('```python\nimport os\nos._exit(0)\n```', spec).correct
    assert not verify('```python\nwhile True:\n    pass\n```', spec).correct


@pytest.mark.skipif(not bwrap_works(), reason='bubblewrap unavailable')
def test_stdio_supports_real_file_descriptors(monkeypatch):
    monkeypatch.setenv('LRR_CODE_SANDBOX', 'bwrap')
    spec = {'kind': 'coding', 'verifier': 'stdio', 'stdio_cases': [['2\n1 2\n', '3\n'], ['1\n' + '9 ' * 40000 + '\n',
                                                                                          str(9 * 40000)]]}
    for program in ('n, *a = map(int, open(0).read().split())\nprint(sum(a))',
                    'import os, sys\nd = sys.stdin.buffer.read().split()\nos.write(1, str(sum(map(int, d[1:]))).encode())',
                    'import sys\nsys.setrecursionlimit(10000)\ninput()\nprint(sum(map(int, input().split())))\nexit()'):
        verdict = verify(f'```python\n{program}\n```', spec)
        assert verdict.correct, (program, verdict.reason)
    reason = verify('```python\nx = [0] * (10 ** 10)\n```', spec).reason
    assert 'MemoryError' in reason or 'signal' in reason


@pytest.mark.skipif(not bwrap_works(), reason='bubblewrap unavailable')
def test_functional_results_are_judged_on_the_host(monkeypatch):
    from docker_code_reward import same_value
    monkeypatch.setenv('LRR_CODE_SANDBOX', 'bwrap')
    assert same_value([1, 2.0000000001], [1, 2]) and not same_value(True, 1) and not same_value([1], [1, 2])
    spec = {'kind': 'coding', 'verifier': 'functional', 'func_name': 'add',
            'cases': [['1\n2', '3'], ['[1, 2]\n0', '[1, 2]'], ['0\n0', 'false']]}
    good = ('class Solution:\n    def add(self, a, b):\n'
            '        return a + [b] * 0 if isinstance(a, list) else (False if a == b == 0 else a + b)')
    assert verify(f'```python\n{good}\n```', spec).correct
    always_equal = ('class E:\n    def __eq__(self, other):\n        return True\n'
                    'class Solution:\n    def add(self, a, b):\n        return E()')
    assert not verify(f'```python\n{always_equal}\n```', spec).correct
    int_subclass = ('class I(int):\n    def __eq__(self, other):\n        return True\n'
                    'class Solution:\n    def add(self, a, b):\n        return I(0)')
    assert not verify(f'```python\n{int_subclass}\n```', spec).correct
    bool_for_int = good.replace('False if', '0 if')
    assert 'case 2' in verify(f'```python\n{bool_for_int}\n```', spec).reason  # 0 is not false
