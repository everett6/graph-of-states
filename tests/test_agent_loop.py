import json

from agent_loop import evaluate_agent_row, run_agent, tool_call, trajectories


def executor_factory(log):
    def executor(code, timeout=10, output_limit=2000):
        log.append(code)
        return {'success': True, 'timed_out': False, 'stdout': str(eval(code.strip().removeprefix('print'))), 'stderr': ''}
    return executor


def scripted(replies):
    def generate(messages, sample):
        return replies.pop(0), 7
    return generate


def test_tool_call_parsing_truncates_after_run():
    code, text = tool_call('Compute.\n<run>\nprint(2**10)\n</run>\nThe output is 999 (hallucinated)')
    assert code.strip() == 'print(2**10)' and text.endswith('</run>')
    assert tool_call('final \\boxed{3}') == (None, 'final \\boxed{3}')
    assert tool_call('<run>\n\n</run>')[0] is None


def test_agent_runs_tools_then_answers_and_verifies():
    log = []
    generate = scripted(['Let me check.\n<run>\nprint(2**10)\n</run>', 'So the answer is \\boxed{1024}.'])
    result = run_agent('What is 2^10?', generate, 3, executor_factory(log))
    assert log == ['print(2**10)\n'] and result['final'].endswith('\\boxed{1024}.')
    assert '1024' in result['messages'][2]['content'] and '2 run(s) left' in result['messages'][2]['content']
    row = {'id': 'r', 'source': 's', 'cutoff_status': 'x', 'prompt': 'What is 2^10?',
           'task_spec': json.dumps({'kind': 'math', 'expected': '1024'})}
    generate = scripted(['<run>\nprint(2**10)\n</run>', '\\boxed{1024}', '\\boxed{1000}'])
    record = evaluate_agent_row(row, generate, 2, 3, executor_factory([]))
    assert [s['correct'] for s in record['samples']] == [True, False] and record['mean_tool_calls'] == 0.5
    rows = list(trajectories([record], [row]))
    assert len(rows) == 1 and rows[0]['messages'][-1]['content'] == '\\boxed{1024}'


def test_tool_budget_is_enforced():
    log = []
    generate = scripted(['<run>\nprint(1)\n</run>'] * 2 + ['<run>\nprint(3)\n</run> \\boxed{3}'])
    result = run_agent('q', generate, 2, executor_factory(log))
    assert len(log) == 2 and 'No runs left' in result['messages'][-2]['content']
    assert result['final'].endswith('\\boxed{3}')  # third reply is final even though it contains <run>
