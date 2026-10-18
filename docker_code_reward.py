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
import threading
import time
import uuid


RUNNER = '''import io, json, os, sys
def main():
    payload = json.load(sys.stdin)
    nonce = payload.pop("nonce")  # local only: an early os._exit(42) from candidate code is not a pass
    report = os.fdopen(os.dup(1), "w")
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    sys.stdout = sys.stderr = io.StringIO()  # candidate prints never reach the report channel
    namespace = {"__name__": "__candidate__"}
    stage = "candidate"
    try:
        exec(compile(payload.pop("code"), "candidate.py", "exec"), namespace)
        stage = "tests"
        exec(compile(payload.pop("tests"), "tests.py", "exec"), namespace)
    except BaseException as error:
        report.write(json.dumps({"stage": stage, "error": type(error).__name__, "message": str(error)[:400]}))
        report.flush()
        os._exit(1)
    report.write("PASS " + nonce)
    report.flush()
    os._exit(42)
main()
'''
FEEDBACK_LIMIT = 2048


def extract_code(completion):
    if isinstance(completion, list):
        completion = '\n'.join(message.get('content', '') for message in completion)
    completion = re.sub(r'<think>.*?</think>', '', completion, flags=re.S).strip()
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', completion, flags=re.S)
    return blocks[-1] if blocks else completion


def code_test_reward(completions, tests=None, **kwargs):
    return [float(passed) for passed, _ in code_test_details(completions, tests)]


BWRAP = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--ro-bind', '/usr', '/usr',
         '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64', '--symlink', 'usr/bin', '/bin',
         '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--clearenv', '--chdir', '/tmp']


def _limits():
    import resource
    for limit, value in ((resource.RLIMIT_AS, 1 << 30), (resource.RLIMIT_CPU, 12),
                         (resource.RLIMIT_FSIZE, 16 << 20), (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(limit, (value, value))


def sandbox_backend():
    """'docker' when the daemon and image are usable, else 'bwrap' (rootless, no network, read-only /usr)."""
    choice = os.environ.get('LRR_CODE_SANDBOX', 'auto')
    image = os.environ.get('LRR_VERIFIER_IMAGE', 'python:3.12-slim')
    if choice in ('auto', 'docker') and shutil.which('docker'):
        inspected = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True, timeout=10)
        if not inspected.returncode:
            return 'docker', image
        if choice == 'docker':
            raise RuntimeError(f'Install verifier image {image!r} separately (and docker access); this verifier never pulls images')
    if choice in ('auto', 'bwrap') and shutil.which('bwrap') and os.path.exists('/usr/bin/python3'):
        return 'bwrap', None
    raise RuntimeError('Code rewards need Docker with the verifier image or bubblewrap (bwrap)')


def code_test_details(completions, tests=None):
    """(passed, feedback) per completion; feedback names the failing stage/exception, bounded."""
    if tests is None or len(tests) != len(completions) or any(not isinstance(t, str) or not t.strip() for t in tests):
        raise ValueError('Each code-reward example needs a nonempty tests string; these datasets do not supply tests automatically')
    backend, image = sandbox_backend()
    outcomes = []
    for completion, assertions in zip(completions, tests):
        name = 'lrr-verifier-' + uuid.uuid4().hex
        nonce = uuid.uuid4().hex
        if backend == 'docker':
            command = ['docker', 'run', '--rm', '--pull=never', '--name', name, '-i',
                '--network=none', '--memory=256m', '--memory-swap=256m', '--cpus=1', '--pids-limit=64',
                '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
                '--user=65534:65534', '--tmpfs=/tmp:rw,noexec,nosuid,size=64m',
                image, 'python', '-I', '-c', RUNNER]
        else:
            command = BWRAP + ['/usr/bin/python3', '-I', '-c', RUNNER]
        process = None
        def kill():
            if backend == 'docker':
                subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            elif process is not None:
                process.kill()
        try:
            # Read at most FEEDBACK_LIMIT bytes; never pipe unbounded candidate output into host RAM.
            # A watchdog kills the sandbox after 15 s, which also unblocks the bounded read.
            watchdog = threading.Timer(15, kill)
            start = time.monotonic()
            try:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, preexec_fn=_limits if backend == 'bwrap' else None)
                watchdog.start()
                with process:
                    try:
                        process.stdin.write(json.dumps({'code': extract_code(completion), 'tests': assertions,
                                                        'nonce': nonce}).encode())
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                    feedback = process.stdout.read(FEEDBACK_LIMIT).decode(errors='replace')
                    process.stdout.close()
                    returncode = process.wait(timeout=20)
            finally:
                watchdog.cancel()
            if time.monotonic() - start >= 15:
                raise subprocess.TimeoutExpired(command, 15)
            if backend == 'docker' and returncode in (125, 126, 127):
                raise RuntimeError('Docker verifier could not start; fix the runtime before training')
            # A candidate exiting early with status 0 is not a passed test run.
            passed = returncode == 42 and feedback == 'PASS ' + nonce
            outcomes.append((passed, '' if passed else (feedback[:FEEDBACK_LIMIT] if not feedback.startswith('PASS')
                                                        else 'forged pass') or f'exit status {returncode}'))
        except subprocess.TimeoutExpired:
            outcomes.append((False, 'timeout after 15 seconds'))
        finally:
            kill()
    return outcomes
