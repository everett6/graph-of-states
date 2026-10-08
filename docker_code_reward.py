"""Optional GRPO unit-test verifier using an already installed Docker image.

No image pulls, host mounts, or network access. Never executes candidate code
in the trainer's Python process. Configure reward_function as
`docker_code_reward:code_test_reward` and supply `tests` in local JSONL rows.
"""
import json
import os
import re
import shutil
import subprocess
import uuid


RUNNER = '''import json, sys
payload = json.load(sys.stdin)
namespace = {"__name__": "__candidate__"}
try:
    exec(compile(payload["code"], "candidate.py", "exec"), namespace)
    exec(compile(payload["tests"], "tests.py", "exec"), namespace)
except BaseException:
    sys.exit(1)
sys.exit(42)
'''


def extract_code(completion):
    if isinstance(completion, list):
        completion = '\n'.join(message.get('content', '') for message in completion)
    completion = re.sub(r'<think>.*?</think>', '', completion, flags=re.S).strip()
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', completion, flags=re.S)
    return blocks[-1] if blocks else completion


def code_test_reward(completions, tests=None, **kwargs):
    if not shutil.which('docker'):
        raise RuntimeError('Code rewards require Docker; exact-answer rewards work without it')
    if tests is None or len(tests) != len(completions) or any(not isinstance(t, str) or not t.strip() for t in tests):
        raise ValueError('Each code-reward example needs a nonempty tests string; these datasets do not supply tests automatically')
    image = os.environ.get('LRR_VERIFIER_IMAGE', 'python:3.12-slim')
    inspected = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True, timeout=10)
    if inspected.returncode:
        raise RuntimeError(f'Install verifier image {image!r} separately; this verifier never pulls images')
    outcomes = []
    for completion, assertions in zip(completions, tests):
        name = 'lrr-verifier-' + uuid.uuid4().hex
        command = ['docker', 'run', '--rm', '--pull=never', '--name', name, '-i',
            '--network=none', '--memory=256m', '--memory-swap=256m', '--cpus=1', '--pids-limit=64',
            '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--user=65534:65534', '--tmpfs=/tmp:rw,noexec,nosuid,size=64m',
            image, 'python', '-I', '-c', RUNNER]
        try:
            # Do not pipe unbounded candidate output into host RAM.
            result = subprocess.run(command, input=json.dumps({'code': extract_code(completion), 'tests': assertions}),
                text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            if result.returncode in (125, 126, 127):
                raise RuntimeError('Docker verifier could not start; fix the runtime before training')
            # A candidate exiting early with status 0 is not a passed test run.
            outcomes.append(float(result.returncode == 42))
        except subprocess.TimeoutExpired:
            outcomes.append(0.0)
        finally:
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10)
    return outcomes
