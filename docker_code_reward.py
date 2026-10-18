"""Optional GRPO unit-test verifier using an already installed Docker image.

No image pulls, host mounts, or network access. Never executes candidate code
in the trainer's Python process. Configure reward_function as
`docker_code_reward:code_test_reward` and supply `tests` in local JSONL rows.
"""
import json
import math
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
# Standard-input programs: the sandbox only returns each run's output; expected outputs never enter it,
# so a candidate cannot forge a pass (the host compares).
STDIO_RUNNER = '''import io, json, os, sys
def main():
    payload = json.load(sys.stdin)
    report = os.fdopen(os.dup(1), "w")
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    sys.stderr = io.StringIO()
    try:
        code = compile(payload["code"], "candidate.py", "exec")
    except BaseException as error:
        report.write(json.dumps({"error": type(error).__name__ + ": " + str(error)[:300]}))
        report.flush()
        os._exit(1)
    outputs = []
    for text in payload["inputs"]:
        raw = io.BytesIO()
        sys.stdin = io.TextIOWrapper(io.BytesIO(text.encode()), encoding="utf-8")
        sys.stdout = io.TextIOWrapper(raw, encoding="utf-8", write_through=True)
        error = None
        try:
            exec(code, {"__name__": "__main__"})
        except SystemExit as stop:
            if stop.code not in (None, 0):
                error = "SystemExit: " + str(stop.code)[:100]
        except BaseException as exc:
            error = type(exc).__name__ + ": " + str(exc)[:300]
        sys.stdout.flush()
        outputs.append({"stdout": raw.getvalue()[:65536].decode(errors="replace"), "error": error})
        if error:
            break
    report.write(json.dumps({"outputs": outputs}))
    report.flush()
    os._exit(0)
main()
'''
FEEDBACK_LIMIT = 2048
STDIO_READ_LIMIT = 4 << 20


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
    outcomes = []
    for completion, assertions in zip(completions, tests):
        nonce = uuid.uuid4().hex
        try:
            returncode, feedback = run_sandboxed(RUNNER, {'code': extract_code(completion), 'tests': assertions,
                                                          'nonce': nonce}, FEEDBACK_LIMIT)
            # A candidate exiting early with status 0 is not a passed test run.
            passed = returncode == 42 and feedback == 'PASS ' + nonce
            outcomes.append((passed, '' if passed else (feedback[:FEEDBACK_LIMIT] if not feedback.startswith('PASS')
                                                        else 'forged pass') or f'exit status {returncode}'))
        except subprocess.TimeoutExpired:
            outcomes.append((False, 'timeout after 15 seconds'))
    return outcomes


def run_sandboxed(script, payload, read_limit, seconds=15):
    """(returncode, bounded stdout) of `python -I -c script` in the sandbox with payload as JSON stdin.
    Raises subprocess.TimeoutExpired after `seconds` (the sandbox is killed)."""
    backend, image = sandbox_backend()
    name = 'lrr-verifier-' + uuid.uuid4().hex
    if backend == 'docker':
        command = ['docker', 'run', '--rm', '--pull=never', '--name', name, '-i',
            '--network=none', '--memory=256m', '--memory-swap=256m', '--cpus=1', '--pids-limit=64',
            '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--user=65534:65534', '--tmpfs=/tmp:rw,noexec,nosuid,size=64m',
            image, 'python', '-I', '-c', script]
    else:
        command = BWRAP + ['/usr/bin/python3', '-I', '-c', script]
    process = None
    def kill():
        if backend == 'docker':
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        elif process is not None:
            process.kill()
    try:
        # Read at most read_limit bytes; never pipe unbounded candidate output into host RAM.
        # A watchdog kills the sandbox after `seconds`, which also unblocks the bounded read.
        watchdog = threading.Timer(seconds, kill)
        start = time.monotonic()
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, preexec_fn=_limits if backend == 'bwrap' else None)
            watchdog.start()
            with process:
                try:
                    process.stdin.write(json.dumps(payload).encode())
                    process.stdin.close()
                except BrokenPipeError:
                    pass
                output = process.stdout.read(read_limit).decode(errors='replace')
                process.stdout.close()
                returncode = process.wait(timeout=seconds + 5)
        finally:
            watchdog.cancel()
        if time.monotonic() - start >= seconds:
            raise subprocess.TimeoutExpired(command, seconds)
        if backend == 'docker' and returncode in (125, 126, 127):
            raise RuntimeError('Docker verifier could not start; fix the runtime before training')
        return returncode, output
    finally:
        kill()


def outputs_match(got, expected):
    """Judge-style comparison: whitespace-separated tokens equal, numbers within 1e-6 (abs or rel)."""
    a, b = got.split(), expected.split()
    if a == b:
        return True
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if not (math.isfinite(fx) and math.isfinite(fy) and math.isclose(fx, fy, rel_tol=1e-6, abs_tol=1e-6)):
            return False
    return True


def stdio_test_details(code, cases):
    """(passed, feedback) for a stdin/stdout program on [[input, expected_output], ...]."""
    if not cases:
        raise ValueError('stdio verifier needs test cases')
    try:
        _, output = run_sandboxed(STDIO_RUNNER, {'code': code, 'inputs': [case[0] for case in cases]},
                                  STDIO_READ_LIMIT)
    except subprocess.TimeoutExpired:
        return False, 'timeout after 15 seconds'
    try:
        report = json.loads(output)
    except ValueError:
        return False, 'program produced no readable result (crashed, killed or exceeded limits)'
    if 'error' in report:
        return False, str(report['error'])[:FEEDBACK_LIMIT]
    runs = report.get('outputs') if isinstance(report.get('outputs'), list) else []
    for index, (stdin, expected) in enumerate(cases):
        run = runs[index] if index < len(runs) and isinstance(runs[index], dict) else None
        if run is None:
            return False, f'case {index}: no result'
        if run.get('error'):
            return False, f'case {index}: input {stdin[:200]!r}: {str(run["error"])[:300]}'
        if not outputs_match(str(run.get('stdout', '')), expected):
            return False, (f'case {index}: input {stdin[:200]!r}: expected {expected[:200]!r}, '
                           f'got {str(run.get("stdout", ""))[:200]!r}')
    return True, ''
