"""Extra modalities, execution traces, INT4 kernels and checkpoint backends."""
import copy
import json
import subprocess
from dataclasses import asdict

import pytest
import torch

from gos_cognition import AsynchronousMatrixBranches
from gos_gemma import attach_gos, frozen_features, selected_logps, TokenGraph, save_gos_adapter, load_gos_adapter
from gos_h2o import h2o_generate, HeavyHitterKV, H2OConfig, h2o_session
from gos_int4 import PackedInt4
from gos_tools import parse_tool_call, gated_tool_turns
from train_unified import GoSSFTTrainer, UnifiedConfig
from unified_data import normalize_row, assistant_tokens, actuator_targets, token_windows, UnifiedCollator
from test_unified import tiny_base, tokenizer
from test_gos_memory import OPTIONS


ALL_OPTIONS = OPTIONS | dict(cross_modal=True, branches=True, async_branches=True, dual_friction=True, tool_gate=True)


def test_complete_head_roundtrip_and_crossmodal_trainable_paths(tmp_path):
    base = tiny_base()
    original = copy.deepcopy(base)
    model = attach_gos(base, 16, 3, 4, ALL_OPTIONS)
    head = model.get_base_model().get_output_embeddings()
    torch.nn.init.normal_(head.lora_B['default'].weight, std=0.02)
    hidden = torch.randn(5, 32)
    states = torch.randn(5, 2, 32)
    mask = torch.ones(5, 2, dtype=torch.bool)
    graph = head.lora_A['default']
    features, risk, diag = graph.features_and_risk(hidden, True, states, mask)
    torch.testing.assert_close(graph(hidden, states, torch.zeros_like(mask)), graph(hidden))
    assert (risk >= diag['semantic_risk']).all()
    assert (risk >= diag['operational_risk']).all()
    with torch.no_grad():
        logits, _ = selected_logps(model, hidden, torch.tensor([2,3,4,5,6]), 2, modal_states=states, modal_mask=mask)
    score, _ = selected_logps(model, hidden, torch.tensor([2,3,4,5,6]), 3, modal_states=states, modal_mask=mask)
    torch.testing.assert_close(score.detach(), logits, atol=1e-6, rtol=1e-5)
    (-score.mean() + diag['coherence_loss'] + risk.mean() + diag['tool_logits'].square().mean()).backward()
    for name, parameter in zip(graph.parameter_names, graph.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    cfg = asdict(UnifiedConfig(graph_width=16, graph_nodes=3, graph_rounds=4, graph_enhancements=ALL_OPTIONS))
    save_gos_adapter(model, tmp_path, cfg)
    restored = load_gos_adapter(original, tmp_path).eval()
    with torch.no_grad():
        expected = model.get_output_embeddings()(hidden, modal_states=states, modal_mask=mask)
        actual = restored.get_output_embeddings()(hidden, modal_states=states, modal_mask=mask)
    torch.testing.assert_close(actual, expected)


def test_tool_trace_targets_and_measured_feedback_reinjection():
    tok = tokenizer()
    messages = normalize_row({'query':'Fix a bug', 'tool_interactions':[
        {'code': 'print(1)', 'sandbox_raw_return': '1', 'success': True}], 'answer':'ok'})
    ids, labels, friction = assistant_tokens(messages, tok)
    targets = actuator_targets(messages, tok, ids, labels)
    assert [value for value in targets if value >= 0] == [1.0, 0.0]
    assert 0.0 in friction
    windows = list(token_windows(ids, labels, friction, 32, 4, {'tool_targets':targets}))
    batch = UnifiedCollator(tok.pad_token_id)(windows)
    assert batch['tool_targets'].ge(0).any()
    call = '<tool_call>' + json.dumps({'name':'python','code':'print(1)'}) + '</tool_call>'
    turns = iter([call, 'fixed'])
    observed = []
    def generate(history):
        observed.append(copy.deepcopy(history))
        return next(turns)
    text, history, records = gated_tool_turns([{'role':'user','content':'task'}], generate, lambda *_:0.99,
        executor=lambda code:{'feedback':'measured output 1', 'success':True})
    assert text == 'fixed' and len(records) == 1
    assert observed[-1][-1]['content'] == '[Tool result]\nmeasured output 1'
    with pytest.raises(ValueError):
        parse_tool_call('<tool_call>{"name":"shell","code":"touch /tmp/x"}</tool_call>')
    called = []
    gated_tool_turns([{'role':'user','content':'task'}], lambda _:call, lambda *_:0.01,
        executor=lambda _:called.append(True))
    assert not called


@pytest.mark.parametrize('width', [7, 32, 65])
def test_int4_error_bounds_and_eviction(width):
    tensor = torch.randn(1, 2, 6, width)
    packed = PackedInt4.pack(tensor)
    restored = packed.unpack()
    # Every value's absolute quantization error is <= half a group scale.
    scales = packed.scales.repeat_interleave(32, -1)[..., :width]
    assert ((restored-tensor).abs() <= scales / 2 + 1e-6).all()
    selected = torch.tensor([[0,3,5],[1,2,5]])
    torch.testing.assert_close(packed.gather_tokens(selected).unpack(), restored.gather(2, selected[None,:,:,None].expand(1,2,3,width)))
    assert PackedInt4.pack(torch.randn(1,2,8,128,dtype=torch.bfloat16)).storage_bytes() < 1*2*8*128*2


@pytest.mark.parametrize('shared', [False, True])
def test_hybrid_sliding_absolute_positions_and_shared_kv(shared):
    from transformers import Gemma4ForCausalLM
    cfg = tiny_base().config
    cfg.sliding_window = 4
    if shared:
        cfg.num_hidden_layers = 4
        cfg.layer_types = ['sliding_attention','full_attention'] * 2
        cfg.num_kv_shared_layers = 2
        # Per-layer properties are built by config initialization.
        values = cfg.to_dict()
        values.pop('per_layer_config', None)
        values.pop('layer_overrides', None)
        values.pop('_heterogeneity_spec', None)
        cfg = type(cfg)(**values)
    model = Gemma4ForCausalLM(cfg).eval()
    ids = torch.tensor([[2,7,8,9,7,8,9,7,8,9]])
    with torch.no_grad():
        expected = model(ids, use_cache=False, logits_to_keep=1).logits
        with h2o_session(model, H2OConfig(16,16)) as session:
            for index in range(ids.shape[1]):
                session['position'] = index
                actual = model(ids[:,index:index+1], position_ids=torch.tensor([[index]]),
                    attention_mask={'full_attention':None,'sliding_attention':None}, use_cache=False, logits_to_keep=1).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    _, stats = h2o_generate(model, ids, 3, heavy_tokens=0, recent_tokens=3, policy='sliding', kv_quantization='int4')
    assert stats['retained_tokens_per_head'] == 3 and stats['cache_policy'] == 'sliding'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_triton_quantization_and_real_async_branches():
    torch.manual_seed(0)  # Regression: BF16 values exactly on half-step boundaries.
    x = torch.randn(1, 3, 5, 65, device='cuda', dtype=torch.bfloat16)
    actual = PackedInt4.pack(x, kernel='triton')
    expected = PackedInt4.pack(x)
    torch.testing.assert_close(actual.unpack(), expected.unpack(), rtol=0, atol=0)
    branches = AsynchronousMatrixBranches(32, 4, asynchronous=True).cuda().eval()
    states = torch.randn(5, 4, 32, device='cuda')
    with torch.no_grad():
        expected = branches(states, allow_async=False)
        actual = branches(states, allow_async=True)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_deepspeed_frozen_input_checkpoint_gradient_subprocess():
    # Keep distributed/Accelerate state isolated from other CPU/GPU tests.
    import sys
    import importlib.util
    if importlib.util.find_spec('deepspeed') is None:
        pytest.skip('Optional DeepSpeed extra is not installed')
    script = '''
import torch, faulthandler
torch.set_num_threads(2)
faulthandler.dump_traceback_later(15)
from gos_runtime import configure_checkpointing
from gos_gemma import attach_gos, selected_logps
from transformers import Gemma4ForCausalLM,Gemma4TextConfig
configure_checkpointing('deepspeed', cpu_offload=True)
config=Gemma4TextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,vocab_size_per_layer_input=32,hidden_size_per_layer_input=8,layer_types=['sliding_attention','full_attention'])
options=dict(scratchpad=True,latent_slots=2,thermal=True,branches=True,async_branches=True,dual_friction=True,tool_gate=True,cross_modal=True)
model=attach_gos(Gemma4ForCausalLM(config).cuda(),16,3,4,options)
head=model.get_output_embeddings();torch.nn.init.normal_(head.lora_B['default'].weight,std=0.02)
hidden=torch.randn(5,32,device='cuda');targets=torch.tensor([2,3,4,5,6],device='cuda')
a,_=selected_logps(model,hidden,targets,2,checkpoint_backend='torch');(-a.mean()).backward()
expected=head.lora_B['default'].weight.grad.clone();model.zero_grad()
b,_=selected_logps(model,hidden,targets,2,checkpoint_backend='deepspeed');(-b.mean()).backward()
torch.testing.assert_close(a,b)
torch.testing.assert_close(head.lora_B['default'].weight.grad,expected)
from gos_runtime import checkpoint_call,cleanup_checkpointing
single=checkpoint_call(lambda x: ((x @ head.lora_B['default'].weight.T).sum(),), (torch.randn(2,16,device='cuda'),), 'deepspeed')
assert isinstance(single,tuple) and len(single)==1
single[0].backward()
from train_unified import GoSSFTTrainer,UnifiedConfig
trainer=object.__new__(GoSSFTTrainer)
trainer.gos_config=UnifiedConfig(graph_width=16,graph_nodes=3,graph_rounds=4,graph_enhancements=options,checkpoint_backend='deepspeed')
ids=torch.tensor([[2,7,8,9,7,8]],device='cuda')
inputs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids.clone(),
    friction_targets=torch.tensor([[-100.,-100.,-100.,1.,-100.,-100.]],device='cuda'),
    tool_targets=torch.tensor([[-100.,-100.,-100.,1.,-100.,0.]],device='cuda'),
    modal_states=torch.randn(1,6,2,32,device='cuda'),modal_mask=torch.ones(1,6,2,device='cuda',dtype=torch.bool))
trainer.compute_loss(model,inputs).backward()
assert torch.isfinite(head.lora_B['default'].weight.grad).all()
cleanup_checkpointing()
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=35)
    assert result.returncode == 0, result.stdout + result.stderr


def test_flash3_guard_on_5070():
    from gos_runtime import validate_attention_backend
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12:
        with pytest.raises(ValueError, match='Hopper'):
            validate_attention_backend('flash_attention_3')


def test_sft_loss_trains_paired_modalities_and_tool_gate():
    model = attach_gos(tiny_base(), 16, 3, 4, ALL_OPTIONS)
    head = model.get_output_embeddings()
    torch.nn.init.normal_(head.lora_B['default'].weight, std=0.02)
    trainer = object.__new__(GoSSFTTrainer)
    trainer.gos_config = UnifiedConfig(graph_width=16, graph_nodes=3, graph_rounds=4, graph_enhancements=ALL_OPTIONS)
    ids = torch.tensor([[2, 7, 8, 9, 7, 8]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
        friction_targets=torch.tensor([[-100., -100., -100., 1., -100., -100.]]),
        tool_targets=torch.tensor([[-100., -100., -100., 1., -100., 0.]]),
        modal_states=torch.randn(1,6,2,32), modal_mask=torch.ones(1,6,2,dtype=torch.bool))
    loss = trainer.compute_loss(model, inputs)
    loss.backward()
    graph = head.lora_A['default']
    grads = {name: parameter.grad for name, parameter in zip(graph.parameter_names, graph.parameters())}
    assert grads['actuator.weight'].abs().sum() > 0
    assert grads['coherence.value.weight'].abs().sum() > 0
    assert grads['operational.2.weight'].abs().sum() > 0


def test_docker_tool_bounds_output_and_keeps_container_restrictions(monkeypatch):
    import io
    import gos_tools
    commands = []
    monkeypatch.setattr(gos_tools.shutil, 'which', lambda _: '/usr/bin/docker')
    monkeypatch.setattr(gos_tools.subprocess, 'run', lambda command, **kwargs: subprocess.CompletedProcess(command, 0))
    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(b'a' * 100000)
            self.stderr = io.BytesIO(b'error detail')
            self.returncode = 42
        def wait(self, timeout):
            return self.returncode
    monkeypatch.setattr(gos_tools.subprocess, 'Popen', Process)
    result = gos_tools.execute_python('print(1)', output_limit=128)
    assert result['success'] and len(result['stdout']) == 128
    assert result['stderr'] == 'error detail'
    assert '--network=none' in commands[0] and '--read-only' in commands[0]
    assert '--pull=never' in commands[0] and '--user=65534:65534' in commands[0]
