"""Symbolic answer equivalence for math rewards; never eval/sympify model text.

LaTeX-ish answers are normalized, tokenized, parsed with `ast` (parse only) and
built into SymPy objects by a whitelist walker. Supports numbers, single-letter
variables, + - * / ^, sqrt/nth roots, pi, i, e-free trig/log/exp, tuples and sets.
Intervals, matrices, inequalities and text answers are rejected (not "wrong").
"""
import ast
import random
import re
import signal
import threading

import sympy

def _factorial(value):
    if not (value.is_Integer and 0 <= value <= 1000):
        raise Unsupported('Factorial needs an integer in [0, 1000]')
    return sympy.factorial(value)


FUNCTIONS = {'factorial': _factorial, 'sqrt': sympy.sqrt, 'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
             'log': sympy.log, 'ln': sympy.log, 'exp': sympy.exp}
CONSTANTS = {'pi': sympy.pi, 'oo': sympy.oo}
MAX_NODES, MAX_EXPONENT, MAX_NUMERIC_EXPONENT, MAX_DIGITS, MAX_LENGTH = 256, 64, 4096, 100, 512
LATEX_WORDS = {r'\pi': ' pi ', r'\infty': ' oo ', r'\cdot': '*', r'\times': '*', r'\div': '/',
               r'\ln': ' ln ', r'\log': ' log ', r'\sin': ' sin ', r'\cos': ' cos ', r'\tan': ' tan ',
               r'\exp': ' exp '}


class Unsupported(ValueError):
    pass


def boxed(text):
    """Content of the last \\boxed{...}/\\fbox{...} with balanced braces, else None."""
    starts = [m.end() for m in re.finditer(r'\\(?:boxed|fbox)\s*\{', text)]
    if not starts:
        return None
    depth, start = 1, starts[-1]
    for index in range(start, len(text)):
        depth += {'{': 1, '}': -1}.get(text[index], 0)
        if depth == 0:
            return text[start:index]
    return None


def _group(text, start):
    """Return (content, end) for a {...} group or single char at text[start]."""
    while start < len(text) and text[start] == ' ':
        start += 1
    if start >= len(text):
        raise Unsupported('Missing argument')
    if text[start] != '{':
        return text[start], start + 1
    depth = 0
    for index in range(start, len(text)):
        depth += {'{': 1, '}': -1}.get(text[index], 0)
        if depth == 0:
            return text[start + 1:index], index + 1
    raise Unsupported('Unbalanced braces')


def _expand(text):
    """Rewrite \\frac and \\sqrt (any nesting) into parenthesized plain math."""
    out, index = [], 0
    while index < len(text):
        match = re.match(r'\\[dt]?frac|\\sqrt', text[index:])
        if not match:
            out.append(text[index])
            index += 1
            continue
        index += match.end()
        if match.group().endswith('frac'):
            top, index = _group(text, index)
            bottom, index = _group(text, index)
            out.append(f'(({_expand(top)})/({_expand(bottom)}))')
        else:
            root = None
            if index < len(text) and text[index] == '[':
                close = text.index(']', index)
                root, index = text[index + 1:close], close + 1
            body, index = _group(text, index)
            out.append(f'(({_expand(body)})**(1/({_expand(root)})))' if root else f'sqrt({_expand(body)})')
    return ''.join(out)


def normalize(text):
    text = str(text).strip().strip('$').strip()
    if len(text) > MAX_LENGTH:
        raise Unsupported('Answer too long')
    text = re.sub(r'\\text\{\s*([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\(?:mathrm|mathbf|operatorname)\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\left|\\right|\\[,!;:]|\\displaystyle|\^\{?\\circ\}?|°|\\%|%|\\\$', '', text)
    text = text.replace(r'\{', '\\SETOPEN ').replace(r'\}', ' \\SETCLOSE')
    text = _expand(text)
    for word, plain in LATEX_WORDS.items():
        text = text.replace(word, plain)
    text = text.replace('\\SETOPEN', '\x01').replace('\\SETCLOSE', '\x02')
    if '\\' in text:
        raise Unsupported('Unsupported LaTeX command')
    text = text.replace('{', '(').replace('}', ')').replace('^', '**')
    if re.fullmatch(r'\s*-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*', text):
        text = text.replace(',', '')  # 1,000 is a number, not a tuple
    text = re.sub(r'(\d+)\s*!', r' factorial(\1) ', text)
    equation = re.fullmatch(r'\s*[A-Za-z]\s*=(.*)', text)
    return equation.group(1) if equation else text


TOKEN = re.compile(r'\s*(\d+\.\d*|\.\d+|\d+|[A-Za-z]+|\*\*|[-+*/(),\x01\x02])')


def _tokens(text):
    tokens, index = [], 0
    while index < len(text):
        match = TOKEN.match(text, index)
        if not match:
            if text[index:].strip():
                raise Unsupported(f'Unexpected character {text[index]!r}')
            break
        index, token = match.end(), match.group(1)
        if token.isalpha() and token not in FUNCTIONS and token not in CONSTANTS:
            tokens.extend(token)  # implicit product of single-letter symbols: "ab" -> a*b
        else:
            tokens.append(token)
    out = []
    for token in tokens:
        if out:
            left, right = out[-1], token
            left_value = left[0].isalnum() or left[0] == '.' or left in (')', '\x02')
            right_value = right[0].isalnum() or right[0] == '.' or right in ('(', '\x01')
            if left_value and right_value and left not in FUNCTIONS:
                out.append('*')
        out.append(token)
    return out


def _build(node, budget):
    budget[0] -= 1
    if budget[0] < 0:
        raise Unsupported('Expression too large')
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        digits = str(node.value)
        if len(digits) > MAX_DIGITS:
            raise Unsupported('Number too long')
        return sympy.Rational(digits)
    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        if node.id == 'i':
            return sympy.I
        if len(node.id) == 1 and node.id.isascii():
            return sympy.Symbol(node.id)
        raise Unsupported(f'Unknown name {node.id}')
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _build(node.operand, budget)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _build(node.left, budget), _build(node.right, budget)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise Unsupported('Division by zero')
            return left / right
        if isinstance(node.op, ast.Pow):
            # Numeric bases may use large integer exponents (3^2025); the bit-size check below bounds them.
            limit = MAX_NUMERIC_EXPONENT if left.is_number and right.is_Integer else MAX_EXPONENT
            if not right.is_number or abs(sympy.N(right)) > limit:
                raise Unsupported('Exponent too large')
            value = left ** right
            if isinstance(value, sympy.Rational) and max(abs(value.p).bit_length(), value.q.bit_length()) > 4096:
                raise Unsupported('Number too large')
            return value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS \
            and len(node.args) == 1 and not node.keywords:
        return FUNCTIONS[node.func.id](_build(node.args[0], budget))
    raise Unsupported('Unsupported expression')


def parse_answer(text):
    """Return ('scalar', expr), ('tuple', [exprs]) or ('set', [exprs])."""
    tokens = _tokens(normalize(text))
    kind = 'scalar'
    if tokens and tokens[0] == '\x01' and tokens[-1] == '\x02':
        kind, tokens = 'set', tokens[1:-1]
    if '\x01' in tokens or '\x02' in tokens:
        raise Unsupported('Nested sets')
    source = ''.join(tokens)
    if not source:
        raise Unsupported('Empty answer')
    tree = ast.parse(source, mode='eval').body
    budget = [MAX_NODES]
    if isinstance(tree, ast.Tuple):
        if kind == 'scalar':
            kind = 'tuple'
        return kind, [_build(item, budget) for item in tree.elts]
    value = _build(tree, budget)
    return (kind, [value]) if kind == 'set' else (kind, value)


class _Deadline:
    """SIGALRM timeout on the main thread; a no-op elsewhere (bounded by MAX_NODES)."""
    def __init__(self, seconds):
        self.seconds = seconds
        self.active = threading.current_thread() is threading.main_thread() and hasattr(signal, 'SIGALRM')

    def __enter__(self):
        if self.active:
            self.previous = signal.signal(signal.SIGALRM, self._fire)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def _fire(self, *_):
        raise TimeoutError('Symbolic check timed out')

    def __exit__(self, *_):
        if self.active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous)


def _scalar_equal(a, b, tolerance):
    difference = a - b
    symbols = sorted(difference.free_symbols, key=str)
    if not symbols:
        value = sympy.N(difference, 60)
        if not value.is_number or value.has(sympy.zoo, sympy.nan):
            return a == b
        return bool(abs(value) <= max(tolerance, sympy.Rational(1, 10**40)))
    if sympy.simplify(difference) == 0:
        return True
    rng = random.Random(0)  # Deterministic sampling: identical verdicts across runs.
    for _ in range(4):
        point = {symbol: sympy.Rational(rng.randint(-97, 97), rng.randint(1, 13)) for symbol in symbols}
        value = sympy.N(difference.subs(point), 60)
        if not value.is_number or abs(value) > sympy.Rational(1, 10**30):
            return False
    return True


def equivalent(actual, reference, tolerance=0, seconds=2.0):
    """True/False for parsed answers; raises Unsupported/TimeoutError if undecidable here."""
    with _Deadline(seconds):
        kind_a, value_a = parse_answer(actual)
        kind_b, value_b = parse_answer(reference)
        tolerance = sympy.Rational(str(tolerance))
        if kind_a == 'scalar' and kind_b == 'scalar':
            return _scalar_equal(value_a, value_b, tolerance)
        if kind_a != kind_b or len(value_a) != len(value_b):
            return False
        if kind_a == 'tuple':
            return all(_scalar_equal(x, y, tolerance) for x, y in zip(value_a, value_b))
        unmatched = list(value_b)
        for item in value_a:
            match = next((index for index, other in enumerate(unmatched)
                          if _scalar_equal(item, other, tolerance)), None)
            if match is None:
                return False
            unmatched.pop(match)
        return True
