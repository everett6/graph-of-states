import copy
import types
import pytest
import torch
from gos_gemma import TokenGraph,attach_gos,set_reasoning_budget
from train_unified import validate_enhancements
from test_unified import tiny_base
from experiment_support import export_gos_gguf,load_gos_gguf


def test_mixed_pass_counts_skip_halted_tokens_and_backpropagate():
    graph=TokenGraph(16,16,3,5,{'adaptive':True,'min_rounds':2,'halting_patience':2,
        'entropy_tolerance':2.,'motion_tolerance':.01})
    calls=[]
    def controlled(self,state,evidence,mask,**kwargs):
        calls.append(len(state))
        delta=(evidence[:,0,0]<0).to(state.dtype)[:,None,None]
        proposal=state+delta
        routes=torch.eye(3)[None].expand(len(state),-1,-1)
        risk=state.new_zeros(state.shape[:2])
        return proposal,risk,routes,risk
    graph._body.transition.forward=types.MethodType(controlled,graph._body.transition)
    # Make the evidence sign deterministic while retaining the normal parameters.
    with torch.no_grad():
        graph._body.project[0].weight.copy_(torch.eye(16));graph._body.project[0].bias.zero_()
    x=torch.zeros(2,16);x[0,0]=1;x[1,0]=-1
    features,_,stats=graph.features_and_risk(x,diagnostics=True)
    assert stats['rounds'].tolist()==[2,5]
    assert calls==[2,2,1,1,1]
    assert stats['converged'].tolist()==[True,False]
    assert stats['hit_budget'].tolist()==[False,True]
    assert stats['executed_token_passes'].item()==7
    features.square().sum().backward()
    assert torch.isfinite(graph.p0.grad).all() and graph.p0.grad.abs().sum()>0


def test_adaptive_full_budget_matches_existing_fixed_path():
    torch.manual_seed(17)
    options={'scratchpad':True,'latent_slots':2}
    fixed=TokenGraph(32,16,3,4,options)
    adaptive=copy.deepcopy(fixed)
    adaptive._body.enhancements=dict(options,adaptive=True,min_rounds=2,halting_patience=2,
        entropy_tolerance=1e-20,motion_tolerance=1e-20)
    x=torch.randn(4,32)
    actual,risk,stats=adaptive.features_and_risk(x,diagnostics=True)
    assert stats['rounds'].tolist()==[4]*4
    torch.testing.assert_close(actual,fixed(x),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(actual,torch.cat([adaptive(row[None]) for row in x]),atol=1e-6,rtol=1e-5)


def test_runtime_policy_preserves_parameters_and_fixed_override():
    model=attach_gos(tiny_base(),16,3,4)
    before={n:p.clone() for n,p in model.named_parameters() if p.requires_grad}
    set_reasoning_budget(model,2,20,True,2)
    graph=model.get_output_embeddings().lora_A['default']
    assert graph._body.rounds==20 and graph._body.enhancements['adaptive']
    for n,p in model.named_parameters():
        if p.requires_grad:torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    set_reasoning_budget(model,3,3,False)
    _,_,stats=graph.features_and_risk(torch.randn(2,32),diagnostics=True)
    assert stats['rounds'].tolist()==[3,3]
    with pytest.raises(ValueError):set_reasoning_budget(model,4,2)
    with pytest.raises(ValueError):validate_enhancements({'adaptive':True,'halting_patience':0},20)
    with pytest.raises(ValueError):validate_enhancements({'adaptive':True,'min_rounds':21},20)


def test_adaptive_gguf_roundtrip(tmp_path):
    base=tiny_base();model=attach_gos(copy.deepcopy(base),16,4,6,
        {'adaptive':True,'min_rounds':2,'halting_patience':2,'entropy_tolerance':2.,'motion_tolerance':100.})
    path=tmp_path/'adaptive.gguf'
    export_gos_gguf(model,path,{'width':16,'rounds':6,'enhancements':model.get_output_embeddings().lora_A['default']._body.enhancements,'base_path':'tiny'})
    restored,_=load_gos_gguf(base,path)
    ids=torch.tensor([[1,2,3]])
    with torch.no_grad():torch.testing.assert_close(model(ids).logits,restored(ids).logits,atol=0,rtol=0)


def test_patience_requires_consecutive_stable_passes():
    graph=TokenGraph(16,16,3,6,{'adaptive':True,'min_rounds':1,'halting_patience':2,
        'entropy_tolerance':2.,'motion_tolerance':.01})
    calls=[]
    def oscillating(self,state,evidence,mask,**kwargs):
        calls.append(len(state));proposal=state+(len(calls)%2)
        risk=state.new_zeros(state.shape[:2]);routes=torch.eye(3)[None].expand(len(state),-1,-1)
        return proposal,risk,routes,risk
    graph._body.transition.forward=types.MethodType(oscillating,graph._body.transition)
    _,_,stats=graph.features_and_risk(torch.randn(2,16),diagnostics=True)
    assert stats['rounds'].tolist()==[6,6] and stats['hit_budget'].all()
