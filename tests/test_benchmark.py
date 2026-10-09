import copy
import pytest
import torch
from test_unified import tiny_base
from experiment_support import (configure,trainable_count,seed_all,save_checkpoint,load_checkpoint,
    export_gos_gguf,load_gos_gguf)
from verified_rewards import verify
from split_audit import grouped_split
from rollout_cache import FrozenFeatureCache
from gos_gemma import inner_lm,frozen_features,TokenGraph


def test_rewards_correct_and_adversarial():
    assert verify(r'\boxed{1/2}',{'kind':'math','expected':'0.5'}).correct
    for text in ('True','__import__("os").system("echo bad")','2**100000','NaN'):
        assert not verify(text,{'kind':'math','expected':'1'}).correct
    task={'kind':'structured','expected':{'a':True}}
    assert verify('{"a":true}',task).correct
    assert not verify('{"a":1}',task).correct
    assert not verify('{"a":true,"a":true}',task).correct
    code={'kind':'coding','verifier':'restricted_python','cases':[{'args':[3],'expected':7}]}
    assert verify('def solve(x):\n return x*2+1',code).correct
    assert not verify('import os\ndef solve(x):\n return 7',code).correct
    assert not verify('def solve(x):\n return x*2',code).correct
    assert not verify('def solve(x):\n return 1',code|{'cases':[{'args':[0],'expected':True}]}).correct


def test_split_near_duplicates_and_trace_groups():
    rows=[{'prompt':'find the value of 1 + 2 and explain every single step carefully today'},
          {'prompt':'find the value of 1 + 2 and explain every single step carefully tomorrow'},
          {'prompt':'unrelated prompt','source':'fable','trace_id':'same'},
          {'prompt':'different window entirely','source':'fable','trace_id':'same'}]
    result,audit=grouped_split(rows,threshold=.75)
    assert result[0]['group']==result[1]['group']
    assert result[2]['group']==result[3]['group']
    assert audit['cross_split_pairs']==0
    reverse,_=grouped_split(list(reversed(rows)),threshold=.75)
    assert [(r['prompt'],r['split']) for r in result]==list(reversed([(r['prompt'],r['split']) for r in reverse]))


def test_matched_parameter_budget_and_initial_equivalence():
    base=tiny_base();ids=torch.tensor([[1,2,3]])
    with torch.no_grad(): reference=base(ids).logits
    gos=configure(copy.deepcopy(base),'gos',16,2,{'scratchpad':True,'latent_slots':2})
    mlp=configure(copy.deepcopy(base),'mlp',16,2,{'scratchpad':True,'latent_slots':2})
    assert trainable_count(gos)==trainable_count(mlp)
    with torch.no_grad():
        torch.testing.assert_close(gos(ids).logits,reference)
        torch.testing.assert_close(mlp(ids).logits,reference)


def test_checkpoint_recovery_exact_next_step(tmp_path):
    seed_all(17);model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    def step():
        optimizer.zero_grad();loss=model(torch.randn(2,3)).square().sum();loss.backward();optimizer.step()
        return loss.detach().clone()
    step();save_checkpoint(tmp_path/'checkpoint.pt',model,optimizer,1,{'seed':17})
    expected=step();parameters=copy.deepcopy(model.state_dict())
    assert load_checkpoint(tmp_path/'checkpoint.pt',model,optimizer,{'seed':17})==1
    torch.testing.assert_close(step(),expected,rtol=0,atol=0)
    for name,p in model.state_dict().items():torch.testing.assert_close(p,parameters[name],rtol=0,atol=0)
    with pytest.raises(ValueError):load_checkpoint(tmp_path/'checkpoint.pt',model,optimizer,{'seed':18})


def test_native_cache_logits_and_feature_lru():
    model=configure(tiny_base(),'gos',16,2);model.eval();ids=torch.tensor([[1,2,3,4]])
    with torch.no_grad():
        full=model(ids,use_cache=False).logits
        cache=None;parts=[]
        for index in range(ids.shape[1]):
            result=model(ids[:,index:index+1],past_key_values=cache,use_cache=True)
            cache=result.past_key_values;parts.append(result.logits)
        torch.testing.assert_close(torch.cat(parts,1),full,atol=2e-6,rtol=2e-5)
    lru=FrozenFeatureCache(1);mask=torch.ones_like(ids)
    a=lru.get(model,ids,mask,4096);b=lru.get(model,ids,mask,4096)
    torch.testing.assert_close(a,b,atol=0,rtol=0);assert lru.hits==1 and lru.misses==1
    lru.get(model,ids[:,:2],mask[:,:2],4096);assert len(lru.items)==1
    next(inner_lm(model).base_model.parameters()).requires_grad_(True)
    with pytest.raises(ValueError):lru.get(model,ids,mask,4096)


def test_gguf_exact_roundtrip(tmp_path):
    pytest.importorskip('gguf');base=tiny_base();model=configure(copy.deepcopy(base),'gos',16,2,{'scratchpad':True})
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:p.add_(.001*torch.randn_like(p))
    path=tmp_path/'adapter.gguf'
    export_gos_gguf(model,path,{'width':16,'rounds':2,'enhancements':{'scratchpad':True},'base_path':'tiny'})
    restored,_=load_gos_gguf(base,path)
    with torch.no_grad():torch.testing.assert_close(model(torch.tensor([[1,2,3]])).logits,
        restored(torch.tensor([[1,2,3]])).logits,rtol=0,atol=0)


def test_default_ablation_equivalence_and_disabled_components():
    graph=TokenGraph(32,16,4,3);explicit=copy.deepcopy(graph)
    explicit._body.enhancements.update(edge_routing=True,auxiliary_critics=True)
    x=torch.randn(3,32)
    torch.testing.assert_close(graph(x),explicit(x),rtol=0,atol=0)
    explicit._body.enhancements.update(edge_routing=False,auxiliary_critics=False)
    features,risk=explicit.features_and_risk(x)
    assert torch.isfinite(features).all();torch.testing.assert_close(risk,torch.zeros_like(risk))


def test_eviction_quality_is_measured_and_bounded():
    from cache_quality import cache_quality
    base=tiny_base();ids=torch.tensor([[1,2,3,4,5,6]])
    exact=cache_quality(base,ids,0,32)
    assert exact['max_logit_error']<2e-6 and exact['evictions']==0
    compressed=cache_quality(base,ids,1,1,'int4')
    assert compressed['evictions']>0
    assert compressed['mean_kl_reference_to_cache']>=-1e-6
    assert 0<=compressed['top1_agreement']<=1


def test_reward_configuration_fails_loudly():
    with pytest.raises(ValueError):verify('1',{'kind':'math','expected':'1','tolerance':'-1'})
    with pytest.raises(ValueError):verify('1',{'kind':'coding','verifier':'typo'})
    assert not verify('True**True',{'kind':'math','expected':'1'}).correct
    task={'kind':'coding','verifier':'restricted_python','cases':[{'args':[0],'expected':[True]}]}
    assert not verify('def solve(x):\n return [1]',task).correct


def test_cached_features_preserve_adapter_gradients():
    from gos_gemma import selected_logps
    model=configure(tiny_base(),'gos',16,2);ids=torch.tensor([[1,2,3]])
    cache=FrozenFeatureCache(2);mask=torch.ones_like(ids)
    cache.get(model,ids,mask,4096)
    features=cache.get(model,ids,mask,4096)
    logps,_=selected_logps(model,features[0,:-1],ids[0,1:],2)
    (-logps.mean()).backward()
    assert model.get_output_embeddings().lora_B['default'].weight.grad.abs().sum()>0
    cache.clear();assert cache.bytes==0 and len(cache.items)==0


def test_task_spec_survives_unified_data(tmp_path):
    import json
    from unified_data import mixed_examples,normalize_row,task_key,task_split
    from test_unified import tokenizer
    row={'prompt':'Return two','answer':'2','task_spec':{'kind':'math','expected':'2'}}
    path=tmp_path/'data.jsonl';path.write_text(json.dumps(row)+'\n')
    split=task_split(task_key(normalize_row(row)),17)
    sources=[{'name':'local','path':str(path),'weight':1}]
    example=next(mixed_examples(sources,tokenizer(),split,mode='grpo',prompt_limit=64))
    assert example['task_spec']==row['task_spec']


def test_actual_trl_grpo_with_native_and_feature_cache(tmp_path,monkeypatch):
    import test_unified as baseline
    original=baseline.UnifiedConfig
    monkeypatch.setattr(baseline,'UnifiedConfig',lambda **kwargs:original(**kwargs,
        grpo_use_cache=True,grpo_feature_cache_entries=4))
    baseline.test_real_trl_grpo_group_and_backward(tmp_path)


def test_local_verified_benchmark_suite_and_manifest(tmp_path):
    import json
    from benchmark_gos import make_dataset
    from test_unified import tokenizer
    row={'prompt':'Return the number two','answer':'2','task_spec':{'kind':'math','expected':'2'}}
    path=tmp_path/'tasks.jsonl';path.write_text(json.dumps(row)+'\n')
    config={'seed':17,'sequence_length':128,'synthetic_tasks':False,'verified_tasks':[str(path)]}
    rows,audit=make_dataset(config,tokenizer())
    assert len(rows)==1 and rows[0]['task_spec']==row['task_spec']
    assert audit['cross_split_pairs']==0
    row['answer']='3';path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError):make_dataset(config,tokenizer())
