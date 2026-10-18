"""Task-specific, fail-closed verifiers; never eval/exec generated host code."""
import ast
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import math
import operator
import re


@dataclass
class Verdict:
    verified: bool
    correct: bool
    reason: str
    def as_dict(self):
        return asdict(self)


def final_text(text):
    if isinstance(text, list):
        text = '\n'.join(message.get('content', '') for message in text)
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()


class RestrictedEvaluator:
    """Arithmetic expressions / a small pure-Python function subset with fuel.

    No host exec/eval, imports, attribute access, I/O, comprehensions, loops,
    or arbitrary calls. Unsupported full Python requires the Docker verifier.
    """
    def __init__(self, budget=1000):
        self.budget = budget

    def expression(self, node, names=None):
        self.budget -= 1
        if self.budget < 0:
            raise ValueError('Evaluation budget exceeded')
        names = names or {}
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            value = Fraction(str(node.value)) if type(node.value) is not bool else node.value
        elif isinstance(node, ast.Name) and node.id in names:
            value = names[node.id]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
            arg = self.expression(node.operand, names)
            value = -arg if isinstance(node.op, ast.USub) else (not arg if isinstance(node.op, ast.Not) else arg)
        elif isinstance(node, ast.BinOp):
            left, right = self.expression(node.left, names), self.expression(node.right, names)
            if not isinstance(left,Fraction) or not isinstance(right,Fraction): raise ValueError('Numeric operands required')
            operations = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
                ast.Div:operator.truediv, ast.FloorDiv:operator.floordiv, ast.Mod:operator.mod, ast.Pow:operator.pow}
            if type(node.op) not in operations:
                raise ValueError('Unsupported operator')
            if isinstance(node.op, ast.Pow) and (right.denominator != 1 or abs(right) > 16):
                raise ValueError('Power budget exceeded')
            value = operations[type(node.op)](left, right)
        elif isinstance(node, ast.Compare) and len(node.ops) == 1:
            operations = {ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge}
            value = operations[type(node.ops[0])](self.expression(node.left,names),self.expression(node.comparators[0],names))
        elif isinstance(node, ast.IfExp):
            value = self.expression(node.body if self.expression(node.test,names) else node.orelse,names)
        elif isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) <= 128:
            value = [self.expression(item,names) for item in node.elts]
        elif isinstance(node, ast.Subscript):
            source, index = self.expression(node.value,names), self.expression(node.slice,names)
            if not isinstance(source,list) or not isinstance(index,Fraction) or index.denominator != 1:
                raise ValueError('Only bounded list indexing is supported')
            value = source[int(index)]
        else:
            raise ValueError('Unsupported expression')
        if isinstance(value,Fraction) and (value.numerator.bit_length() > 512 or value.denominator.bit_length() > 512):
            raise ValueError('Numeric size budget exceeded')
        return value

    def function(self, code, name, arguments):
        if len(code) > 16384:
            raise ValueError('Code too long')
        module = ast.parse(code)
        if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
            raise ValueError('Expected one pure function')
        function = module.body[0]
        if function.args.posonlyargs or len({arg.arg for arg in function.args.args}) != len(function.args.args): raise ValueError('Unsupported arguments')
        if function.name != name or function.decorator_list or function.args.defaults or function.args.kwonlyargs or function.args.vararg or function.args.kwarg:
            raise ValueError('Unsupported function signature')
        if len(function.args.args) != len(arguments):
            raise ValueError('Wrong argument count')
        def convert(value):
            if type(value) is bool: return value
            if type(value) in (int,float): return Fraction(str(value))
            if isinstance(value,list) and len(value)<=128: return [convert(item) for item in value]
            raise ValueError('Unsupported argument')
        names = {arg.arg:convert(value) for arg,value in zip(function.args.args,arguments)}
        def block(statements):
            for statement in statements:
                if isinstance(statement,ast.Return): return True,self.expression(statement.value,names)
                if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
                    names[statement.targets[0].id]=self.expression(statement.value,names)
                elif isinstance(statement,ast.If):
                    returned,value=block(statement.body if self.expression(statement.test,names) else statement.orelse)
                    if returned: return True,value
                elif not isinstance(statement,ast.Assign): raise ValueError('Unsupported statement')
                else: raise ValueError('Unsupported assignment')
            return False,None
        returned,result=block(function.body)
        if not returned: raise ValueError('Missing return')
        return result


def answer_text(completion):
    """The final answer span: last balanced \\boxed{}, else the last line minus an 'Answer:' prefix."""
    from symbolic_math import boxed
    text = final_text(completion)
    if len(text)>65536: raise ValueError('Answer too long')
    inner = boxed(text)
    if inner is None:
        lines = text.strip().splitlines()
        inner = lines[-1] if lines else ''
    return re.sub(r'^(?:final answer|answer)\s*:\s*', '', inner.strip(), flags=re.I).strip()


def numeric_value(text):
    value=RestrictedEvaluator().expression(ast.parse(str(text).replace('^','**'),mode='eval').body)
    if type(value) is bool or not isinstance(value,Fraction): raise ValueError('Expected numeric answer')
    return value


def math_reward(completion, expected, tolerance='0'):
    """Exact rational check when both sides are plain numbers; otherwise SymPy equivalence."""
    from symbolic_math import Unsupported, equivalent, parse_answer
    tol=Fraction(str(tolerance))
    if tol<0: raise ValueError('Invalid numeric reference/tolerance')
    try:
        reference=numeric_value(expected)
    except (ValueError,SyntaxError,TypeError,ZeroDivisionError,OverflowError,RecursionError):
        reference=None
        try: parse_answer(expected)
        except (Unsupported,SyntaxError,TypeError,ValueError,ZeroDivisionError) as error:
            raise ValueError('Invalid math reference') from error
    try:
        text = answer_text(completion)
        if reference is not None:
            try:
                return Verdict(True,abs(numeric_value(text)-reference)<=tol,'numeric_equivalence')
            except (ValueError,SyntaxError,TypeError,KeyError,ZeroDivisionError,IndexError,OverflowError,RecursionError):
                pass
        return Verdict(True,bool(equivalent(text,expected,tol)),'symbolic_equivalence')
    except (Unsupported,TimeoutError,ValueError,SyntaxError,TypeError,KeyError,ZeroDivisionError,IndexError,
            OverflowError,RecursionError,AttributeError):
        return Verdict(True,False,'invalid_math_answer')


def strict_json(text):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    def finite_float(text):
        value=float(text)
        if not math.isfinite(value): raise ValueError('Nonfinite JSON number')
        return value
    return json.loads(text,object_pairs_hook=pairs,parse_float=finite_float,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def structured_reward(completion, expected, schema=None):
    if schema is not None:
        import jsonschema
        jsonschema.Draft202012Validator.check_schema(schema)
    try:
        text=final_text(completion)
        if len(text)>65536: raise ValueError('Answer too long')
        fenced=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',text,re.S)
        actual=strict_json(fenced.group(1) if fenced else text)
        if schema is not None:
            jsonschema.validate(actual,schema)
        def equal(a,b):
            if type(a) is bool or type(b) is bool: return type(a) is type(b) and a==b
            if isinstance(a,dict) and isinstance(b,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,list) and isinstance(b,list): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return type(a) is type(b) and a==b
        return Verdict(True,equal(actual,expected),'schema_and_reference')
    except Exception as error:
        # Invalid references/configuration must not silently become model errors.
        if type(error).__name__=='SchemaError': raise
        return Verdict(True,False,'invalid_structured_answer')


def coding_reward(completion, spec):
    if spec.get('verifier','docker') not in ('docker','restricted_python'): raise ValueError('Unknown code verifier')
    from docker_code_reward import extract_code,code_test_details
    code=extract_code(final_text(completion))
    if spec.get('verifier','docker')=='restricted_python':
        try:
            cases=spec['cases']
            if not cases: raise ValueError('Missing test cases')
            for case in cases:
                actual=RestrictedEvaluator().function(code,spec.get('function','solve'),case['args'])
                def expected(value):
                    if type(value) is bool: return value
                    if isinstance(value,list): return [expected(item) for item in value]
                    return Fraction(str(value))
                target=expected(case['expected'])
                if (type(actual) is bool) != (type(target) is bool): return Verdict(True,False,'wrong_type')
                def equal(a,b):
                    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
                    if (type(a) is bool)!=(type(b) is bool):return False
                    return a==b
                if not equal(actual,target): return Verdict(True,False,'unit_test_failure')
            return Verdict(True,True,'restricted_python_tests')
        except (ValueError,SyntaxError,KeyError,TypeError,ZeroDivisionError,IndexError,OverflowError,RecursionError):
            return Verdict(True,False,'restricted_python_rejection')
    try:
        (passed,feedback),=code_test_details([code],tests=[spec['tests']])
        return Verdict(True,passed,'container_tests' if passed else 'container_test_failure: '+feedback)
    except (RuntimeError,PermissionError) as error:
        return Verdict(False,False,'verifier_unavailable: '+str(error))


def verify(completion, task):
    task=_spec(task)
    kind=task['kind']
    if kind=='math': return math_reward(completion,task['expected'],task.get('tolerance','0'))
    if kind=='structured': return structured_reward(completion,task['expected'],task.get('schema'))
    if kind=='coding': return coding_reward(completion,task)
    if kind=='lean': return lean_reward(completion,task)
    raise ValueError('Unknown task kind')


def lean_reward(completion, spec):
    """Lean 4 + Mathlib check of the tactic proof against the exact dataset statement."""
    from lean_verifier import DEFAULT_OPENS, check_proof, extract_proof, feedback
    try:
        result=check_proof(spec['statement'],extract_proof(completion),spec.get('opens',DEFAULT_OPENS))
    except RuntimeError as error:
        return Verdict(False,False,'verifier_unavailable: '+str(error))
    return Verdict(True,result['correct'],'lean_checked' if result['correct'] else 'lean_failure: '+feedback(result))


def task_verified_reward(completions, task_spec, **kwargs):
    if len(completions)!=len(task_spec): raise ValueError('Task/reward batch mismatch')
    verdicts=[verify(text,spec) for text,spec in zip(completions,task_spec)]
    if any(not verdict.verified for verdict in verdicts): raise RuntimeError('A task verifier is unavailable')
    return [float(verdict.correct) for verdict in verdicts]


def _spec(value):
    """Task specs may arrive as JSON strings (Arrow-safe) or dicts padded with None by Arrow."""
    if isinstance(value,str): value=json.loads(value)
    return None if value is None else {key:item for key,item in value.items() if item is not None}


def grpo_verified_reward(completions, answer=None, task_spec=None, **kwargs):
    """GRPO reward: task_spec verifier when present, else SymPy check of the reference's
    \\boxed{} answer, else exact final-text match (the old baseline). Fails loudly when a
    configured verifier is unavailable."""
    from symbolic_math import boxed
    answers=answer if answer is not None else [None]*len(completions)
    specs=task_spec if task_spec is not None else [None]*len(completions)
    if not len(completions)==len(answers)==len(specs): raise ValueError('Task/reward batch mismatch')
    rewards=[]
    for completion,reference,raw in zip(completions,answers,specs):
        spec=_spec(raw)
        if spec:
            verdict=verify(completion,spec)
            if not verdict.verified: raise RuntimeError('A task verifier is unavailable: '+verdict.reason)
            rewards.append(float(verdict.correct)); continue
        if reference is None: raise ValueError('Row has neither task_spec nor reference answer')
        target=boxed(final_text(reference))
        if target is not None:
            try:
                rewards.append(float(math_reward(completion,target).correct)); continue
            except ValueError:
                pass  # Unparseable reference answer: fall back to exact text.
        rewards.append(float(final_text(completion)==final_text(reference)))
    return rewards


def random_control_reward(completions, **kwargs):
    """Control: a fair coin keyed on the completion text, independent of correctness.
    A GRPO gain under this reward is not evidence that correctness feedback helped."""
    import hashlib
    return [float(hashlib.sha256(('gos-random-control-v1:'+final_text(text)).encode()).digest()[0]&1)
            for text in completions]


def format_only_reward(completions, answer=None, task_spec=None, **kwargs):
    """Control: 1 when the answer is in checkable form (parseable \\boxed{} for math, a Python block that
    parses for code), regardless of correctness."""
    from docker_code_reward import extract_code
    from symbolic_math import Unsupported, boxed, parse_answer
    answers=answer if answer is not None else [None]*len(completions)
    specs=task_spec if task_spec is not None else [None]*len(completions)
    rewards=[]
    for completion,reference,raw in zip(completions,answers,specs):
        spec=_spec(raw) or {}
        text=final_text(completion)
        if spec.get('kind')=='coding':
            try: ast.parse(extract_code(text)); rewards.append(1.0)
            except SyntaxError: rewards.append(0.0)
            continue
        if spec.get('kind')=='math' or (reference is not None and boxed(final_text(reference)) is not None):
            inner=boxed(text)
            try: ok=inner is not None and parse_answer(inner) is not None
            except (Unsupported,SyntaxError,TypeError,ValueError,ZeroDivisionError): ok=False
            rewards.append(float(ok)); continue
        rewards.append(float(bool(text.strip())))
    return rewards
