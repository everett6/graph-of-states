"""Machine-checked proofs: compile a candidate Lean 4 proof against Mathlib in a sandbox.

A proof counts only if the file elaborates with no errors, the target theorem is the exact
dataset statement (placed before any model text), and `#print axioms` shows nothing beyond
propext / Classical.choice / Quot.sound (so `sorry`, `native_decide`, new axioms, `#exit`
tricks and `unsafe`/`implemented_by` escapes all fail). Runs under bubblewrap: no network,
read-only toolchain and Mathlib, private /tmp.
"""
from functools import lru_cache
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

PROJECT = Path(os.environ.get('GOS_LEAN_PROJECT', Path(__file__).parent / 'lean_env')).resolve()
ELAN = Path(os.environ.get('ELAN_HOME', Path.home() / '.elan'))
ALLOWED_AXIOMS = {'propext', 'Classical.choice', 'Quot.sound'}
BANNED = re.compile(r'\b(sorry|admit|axiom|unsafe|implemented_by|extern|opaque|native_decide|decide\s*:=\s*true'
                    r'|run_cmd|run_tac|#exit|#eval|elab|macro|syntax|initialize|builtin_initialize)\b|set_option\s+debug')
TARGET = '__gos_target'
DEFAULT_OPENS = 'open BigOperators Real Nat Topology Rat'


@lru_cache(maxsize=1)
def lean_environment():
    toolchain = (PROJECT / 'lean-toolchain').read_text().strip().replace('/', '--').replace(':', '---')
    lean = ELAN / 'toolchains' / toolchain / 'bin' / 'lean'
    if not lean.exists():
        raise RuntimeError(f'Lean toolchain missing at {lean}; run elan in {PROJECT}')
    path = subprocess.run([str(ELAN / 'bin' / 'lake'), 'env', 'printenv', 'LEAN_PATH'], cwd=PROJECT,
                          capture_output=True, text=True, timeout=120, check=True).stdout.strip()
    return str(lean), path


def normalize_statement(formal_statement):
    """Split a dataset statement into (statement, open lines): drop imports/set_option and doc comments
    (the verifier supplies `import Mathlib` and fixed options), move `open ...` lines out."""
    text = re.sub(r'/-[-!]?.*?-/', '', formal_statement, flags=re.S)
    opens, kept = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('set_option '):
            continue
        if stripped.startswith('open ') and not any(item.strip() for item in kept):
            opens.append(stripped)
            continue
        kept.append(line)
    # Mathlib renamed big-operator binders `∑ x in s, f` -> `∑ x ∈ s, f` (same meaning).
    statement = re.sub(r'([∑∏]\s*[^,∈∑∏]+?)\s+in\s+', r'\1 ∈ ', '\n'.join(kept).strip())
    return statement, '\n'.join(opens)


def statement_body(formal_statement):
    """'[defs] theorem name args : type := [by] [sorry]' -> (name, 'args : type', defs).

    Dataset definitions before the theorem are kept verbatim (they precede all model text)."""
    text = re.sub(r':=\s*(by)?\s*(sorry)?\s*$', '', formal_statement.strip())
    starts = [m.start() for m in re.finditer(r'^(?:theorem|lemma)\s', text, re.M)]
    if not starts:
        raise ValueError('Expected a theorem statement')
    preamble, theorem = text[:starts[-1]].strip(), text[starts[-1]:]
    match = re.fullmatch(r'(?:theorem|lemma)\s+([^\s(\[{:]+)\s*(.*)', theorem, re.S)
    if not match or not match.group(2).strip():
        raise ValueError('Expected a single theorem statement')
    return match.group(1), match.group(2).strip(), preamble


def build_file(formal_statement, proof, opens=DEFAULT_OPENS):
    """The target theorem is renamed and fixed by us; model text is only the tactic block."""
    _, body, preamble = statement_body(formal_statement)
    tactics = '\n'.join('  ' + line for line in proof.strip('\n').splitlines()) or '  skip'
    preamble = preamble + '\n\n' if preamble else ''
    return (f'import Mathlib\n{opens}\n\n{preamble}theorem {TARGET} {body} := by\n{tactics}\n\n'
            f'#print axioms {TARGET}\n')


def extract_proof(completion):
    """Tactic block from the last ```lean block (or raw text); strips a restated theorem header."""
    from verified_rewards import final_text
    text = final_text(completion)
    blocks = re.findall(r'```(?:lean4?|Lean)?\s*\n(.*?)```', text, re.S)
    proof = blocks[-1] if blocks else text
    header = re.search(r'(?:theorem|lemma)\s.*?:=\s*by\b', proof, re.S)
    if header:
        proof = proof[header.end():]
    lines = proof.strip('\n').splitlines()
    indent = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
    return '\n'.join(line[indent:] for line in lines)


def run_lean(source, timeout=120, limit=20000):
    """(returncode, last `limit` chars of combined output) for one Lean file, sandboxed and time-limited."""
    lean, lean_path = lean_environment()
    with tempfile.TemporaryDirectory(prefix='gos-lean-') as directory:
        path = Path(directory) / 'Check.lean'
        path.write_text(source)
        command = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session',
                   '--ro-bind', '/usr', '/usr', '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64',
                   '--symlink', 'usr/bin', '/bin', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
                   '--ro-bind', str(ELAN), str(ELAN), '--ro-bind', str(PROJECT), str(PROJECT),
                   '--ro-bind', str(path), '/work/Check.lean', '--clearenv', '--setenv', 'LEAN_PATH', lean_path,
                   '--setenv', 'HOME', '/tmp', '--chdir', '/work', lean, '/work/Check.lean']
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            return result.returncode, (result.stdout + result.stderr)[-limit:]
        except subprocess.TimeoutExpired:
            return -1, f'timeout after {timeout} seconds'


def check_proof(formal_statement, proof, opens=DEFAULT_OPENS, timeout=120):
    """{'correct', 'reason', 'messages', 'seconds'}; reason explains every failure."""
    start = time.perf_counter()
    banned = BANNED.search(proof)
    if banned:
        return {'correct': False, 'reason': f'banned construct: {banned.group(0)}', 'messages': '', 'seconds': 0.0}
    source = build_file(formal_statement, proof, opens)
    offset = source.split(f'theorem {TARGET}', 1)[0].count('\n') + source.split(f'theorem {TARGET}', 1)[1].split(':= by\n', 1)[0].count('\n') + 1  # file lines before the first tactic line
    code, output = run_lean(source, timeout)
    seconds = time.perf_counter() - start
    errors = [line for line in output.splitlines() if ': error' in line or line.startswith('error')]
    audit = re.search(rf"'{TARGET}' (?:depends on axioms: \[(.*?)\]|does not depend on any axioms)", output, re.S)
    if code == -1:
        return {'correct': False, 'reason': 'timeout', 'messages': output, 'seconds': seconds, 'offset': offset}
    if errors or code != 0:
        return {'correct': False, 'reason': 'lean_error', 'messages': output, 'seconds': seconds, 'offset': offset}
    if not audit:
        return {'correct': False, 'reason': 'axiom audit missing', 'messages': output, 'seconds': seconds, 'offset': offset}
    axioms = {name.strip() for name in (audit.group(1) or '').split(',') if name.strip()}
    if axioms - ALLOWED_AXIOMS:
        return {'correct': False, 'reason': f'disallowed axioms: {sorted(axioms - ALLOWED_AXIOMS)}',
                'messages': output, 'seconds': seconds, 'offset': offset}
    return {'correct': True, 'reason': 'lean_checked', 'messages': output, 'seconds': seconds, 'offset': offset}


# Text that can change the environment for later theorems in a shared file (new commands, attributes,
# an unterminated comment). Failures after such a proof are rechecked alone.
LEAKY = re.compile(r'^\s*(?:@\[|attribute\b|instance\b|namespace\b|section\b|end\b|open\b|universe\b|variable\b'
                   r'|theorem\b|lemma\b|def\b|abbrev\b|example\b|notation\b|infix|prefix|postfix|local\b|scoped\b'
                   r'|set_option\b|#)|/-', re.M)
CHUNK = re.compile(r'^/work/Check\.lean:(\d+):\d+: ', re.M)


def _batch_source(statements, proofs, opens, preamble):
    """One Lean file with theorem __gos_target_i per candidate; returns (source, [(first, last) line])."""
    head = f'import Mathlib\n{opens}\n\n' + (preamble + '\n\n' if preamble else '')
    parts, ranges, line = [head], [], head.count('\n') + 1
    for index, (statement, proof) in enumerate(zip(statements, proofs)):
        _, body, _ = statement_body(statement)
        tactics = '\n'.join('  ' + item for item in proof.strip('\n').splitlines()) or '  skip'
        block = f'theorem {TARGET}_{index} {body} := by\n{tactics}\n\n#print axioms {TARGET}_{index}\n\n'
        offset = line + body.count('\n')  # file line of ':= by'; tactic line k is file line offset + k
        ranges.append((line, line + block.count('\n') - 1, offset))
        parts.append(block)
        line += block.count('\n')
    return ''.join(parts), ranges


def check_proofs(statements, proofs, opens=DEFAULT_OPENS, timeout=120, confirm=True, batch=32):
    """check_proof for many candidates with one Mathlib import per group (import dominates the cost).

    Same verdicts as checking each alone: a batch pass is only a candidate and is confirmed by
    check_proof in its own file; failures are rechecked alone when another proof could have
    interfered (environment-changing text, errors outside every theorem, timeout, missing audit).
    confirm=False accepts a batch pass with a clean per-theorem axiom audit and no errors: only for
    filtering non-adversarial training data, never for scoring a model."""
    if isinstance(statements, str):
        statements = [statements] * len(proofs)
    results = [None] * len(proofs)
    groups = {}
    for index, (statement, proof) in enumerate(zip(statements, proofs)):
        banned = BANNED.search(proof)
        if banned:
            results[index] = {'correct': False, 'reason': f'banned construct: {banned.group(0)}', 'messages': '',
                              'seconds': 0.0}
        else:
            groups.setdefault(statement_body(statement)[2], []).append(index)
    chunks_of = [(preamble, indices[i:i + batch]) for preamble, indices in groups.items()
                 for i in range(0, len(indices), batch)]
    for preamble, indices in chunks_of:
        if len(indices) == 1:
            results[indices[0]] = check_proof(statements[indices[0]], proofs[indices[0]], opens, timeout)
            continue
        start = time.perf_counter()
        source, ranges = _batch_source([statements[i] for i in indices], [proofs[i] for i in indices], opens, preamble)
        code, output = run_lean(source, timeout + 15 * len(indices), limit=1 << 24)  # every theorem's audit
        if code == -1:  # one slow proof stalls the file: bisect instead of checking every candidate alone
            half = len(indices) // 2
            for part in (indices[:half], indices[half:]):
                for index, result in zip(part, check_proofs([statements[i] for i in part], [proofs[i] for i in part],
                                                            opens, timeout, confirm, batch)):
                    results[index] = result
            continue
        seconds = (time.perf_counter() - start) / len(indices)
        marks = list(CHUNK.finditer(output))
        chunks = [(int(m.group(1)), output[m.start():(nxt.start() if nxt else len(output))])
                  for m, nxt in zip(marks, marks[1:] + [None])]
        stray = [text for number, text in chunks if ': error' in text.split('\n', 1)[0]
                 and not any(first <= number <= last for first, last, _ in ranges)]
        trusted = code != -1 and not stray
        for position, (index, (first, last, offset)) in enumerate(zip(indices, ranges)):
            mine = [text for number, text in chunks if first <= number <= last]
            errors = [text for text in mine if ': error' in text.split('\n', 1)[0]]
            clean_before = not any(LEAKY.search(proofs[i]) for i in indices[:position])
            if errors and trusted and clean_before:
                results[index] = {'correct': False, 'reason': 'lean_error', 'messages': ''.join(mine),
                                  'seconds': seconds, 'offset': offset}
                continue
            audit = re.search(rf"'{TARGET}_{position}' (?:depends on axioms: \[(.*?)\]|does not depend on any axioms)",
                              output, re.S)
            axioms = {name.strip() for name in ((audit.group(1) or '') if audit else '').split(',') if name.strip()}
            if not confirm and trusted and not errors and audit and axioms <= ALLOWED_AXIOMS:
                results[index] = {'correct': True, 'reason': 'lean_checked_batch', 'messages': '',
                                  'seconds': seconds, 'offset': offset}
            else:  # candidate pass, or a verdict another proof may have affected: decide it alone
                results[index] = check_proof(statements[index], proofs[index], opens, timeout)
    return results


def feedback(result, limit=1500):
    """Compact error text for a repair turn: error lines with their following context."""
    if result['correct']:
        return 'Proof checked.'
    if result['reason'] != 'lean_error':
        return result['reason']
    offset = result.get('offset', 0)
    text = re.sub(r'/work/Check\.lean:(\d+):(\d+)', lambda m: f'proof line {int(m.group(1)) - offset}:{m.group(2)}',
                  result['messages'])
    return text[:limit]


def statement_compiles(formal_statement, opens=DEFAULT_OPENS, timeout=120):
    """True when the statement (and its dataset definitions) elaborates on this Mathlib, proof as sorry."""
    code, output = run_lean(build_file(formal_statement, 'sorry', opens), timeout)
    return code == 0 and not any(': error' in line for line in output.splitlines())
