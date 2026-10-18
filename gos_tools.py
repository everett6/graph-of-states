"""Actuator-gated Python tool turns; generated code runs only in a sandbox (Docker or bubblewrap)."""
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import uuid


# Optional read-only math libraries for tool runs (scripts/build_sandbox_libs.sh: sympy, mpmath).
TOOL_LIBS = Path(os.environ.get('GOS_TOOL_LIBS', Path(__file__).resolve().parent / 'sandbox_libs'))
SANDBOX_LIBS = '/opt/gos-libs'

RUNNER = '''import json, os, sys, traceback
if os.path.isdir('/opt/gos-libs'):
    sys.path.insert(0, '/opt/gos-libs')
payload = json.load(sys.stdin)
try:
    exec(compile(payload['code'], 'candidate.py', 'exec'), {'__name__':'__candidate__'})
except BaseException:
    traceback.print_exc()
    sys.exit(1)
sys.exit(42)
'''


def parse_tool_call(text):
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)
    if not matches:
        return None
    call = json.loads(matches[-1])
    if not isinstance(call, dict) or call.get('name') != 'python' or not isinstance(call.get('code'), str):
        raise ValueError('Only structured python code tool calls are executable; shell commands are rejected')
    if not call['code'].strip() or len(call['code'].encode()) > 65536:
        raise ValueError('Tool code must contain 1..65536 bytes')
    return call


def tool_libs_available():
    """True when sympy/mpmath can be mounted into the bubblewrap tool sandbox (not into Docker)."""
    return (TOOL_LIBS / 'sympy').is_dir() and (TOOL_LIBS / 'mpmath').is_dir()


def execute_python(code, timeout=10, output_limit=16384, image=None):
    """Return measured execution success and bounded raw feedback, not correctness."""
    if not isinstance(code, str) or not code.strip() or len(code.encode()) > 65536:
        raise ValueError('Tool code must contain 1..65536 bytes')
    if timeout <= 0 or output_limit < 1:
        raise ValueError('Tool timeout and output limit must be positive')
    from docker_code_reward import BWRAP, _limits, sandbox_backend
    if image is not None:
        os.environ.setdefault('LRR_VERIFIER_IMAGE', image)
    backend, image = sandbox_backend()  # Docker when usable, else rootless bubblewrap
    name = 'gos-tool-' + uuid.uuid4().hex
    if backend == 'docker':
        command = ['docker', 'run', '--rm', '--pull=never', '--name', name, '-i', '--network=none',
            '--memory=256m', '--memory-swap=256m', '--cpus=1', '--pids-limit=64', '--read-only',
            '--cap-drop=ALL', '--security-opt=no-new-privileges', '--user=65534:65534',
            '--tmpfs=/tmp:rw,noexec,nosuid,size=64m', image, 'python', '-I', '-c', RUNNER]
    else:
        libs = ['--ro-bind', str(TOOL_LIBS), SANDBOX_LIBS] if tool_libs_available() else []
        command = BWRAP + libs + ['/usr/bin/python3', '-I', '-c', RUNNER]
    process = None
    threads = []
    buffers = [bytearray(), bytearray()]
    timed_out = False
    def drain(stream, buffer):
        try:
            while block := stream.read(4096):
                buffer.extend(block[:max(0, output_limit - len(buffer))])
        finally:
            stream.close()
    def write_payload():
        try:
            process.stdin.write(json.dumps({'code': code}).encode())
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   preexec_fn=_limits if backend == 'bwrap' else None)
        for stream, buffer in zip((process.stdout, process.stderr), buffers):
            thread = threading.Thread(target=drain, args=(stream, buffer), daemon=True)
            thread.start(); threads.append(thread)
        writer = threading.Thread(target=write_payload, daemon=True)
        writer.start(); threads.append(writer)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill(); process.wait(timeout=5)
        if backend == 'docker' and process.returncode in (125, 126, 127):
            raise RuntimeError('Docker tool could not start')
    finally:
        if backend == 'docker':
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        for thread in threads:
            thread.join(timeout=1)
    stdout, stderr = (bytes(buffer).decode(errors='replace') for buffer in buffers)
    success = not timed_out and process.returncode == 42
    return {'success': success, 'failure': float(not success), 'stdout': stdout, 'stderr': stderr,
            'timed_out': timed_out, 'returncode': process.returncode,
            'feedback': 'TimeoutError: tool execution deadline exceeded' if timed_out else stdout + '\n' + stderr}


def gated_tool_turns(messages, generate, gate, threshold=0.85, max_turns=4, executor=execute_python):
    """Pause between decoded tool turns and reinject measured output as context.

    `generate(messages)` returns text; `gate(messages, text)` returns a learned
    probability. No external side effects occur inside autograd or checkpoint
    recomputation. A latent vector alone is not an executable program.
    """
    if not 0 <= threshold <= 1 or max_turns < 1:
        raise ValueError('Invalid tool threshold/turn budget')
    history = [dict(item) for item in messages]
    records = []
    for index in range(max_turns):
        text = generate(history)
        score = float(gate(history, text))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Tool gate must return a finite probability in [0, 1]')
        call = parse_tool_call(text)
        history.append({'role': 'assistant', 'content': text})
        if call is None or score < threshold:
            return text, history, records
        if index + 1 == max_turns:
            # Avoid running a tool whose result cannot be consumed by the model.
            return text, history, records
        outcome = executor(call['code'])
        records.append({'gate_probability': score, **outcome})
        history.append({'role': 'user', 'content': '[Tool result]\n' + outcome['feedback']})
    return text, history, records
