"""Offline, versioned controlled training/evaluation. Each variant is a fresh process."""
import argparse, contextlib, hashlib, importlib.metadata, json, random, subprocess, sys, time
from pathlib import Path
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
from torch.utils.checkpoint import checkpoint
from experiment_support import configure,seed_all,trainable_count,save_checkpoint,load_checkpoint,export_gos_gguf
from gos_gemma import inner_lm,selected_logps
from local_gemma import load_local_gemma
from split_audit import grouped_split
from verified_rewards import verify, verify_many


def write(path,value): Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def make_dataset(config,tokenizer):
    rows=[]
    for source in config.get('local_data',[]):
        count=0
        with open(source) as stream:
            for line in stream:
                row=json.loads(line)
                if row.get('window_index',0)!=0: continue
                ids=row['input_ids'];labels=row['labels']
                active=[i for i,v in enumerate(labels) if v!=-100]
                if not active or active[0]>=config['sequence_length']-8: continue
                rows.append({'prompt':tokenizer.decode(ids[:active[0]]),'input_ids':ids[:config['sequence_length']],
                    'labels':labels[:config['sequence_length']],'source':row.get('source',str(Path(source).parent)),
                    'trace_id':row.get('trace_id',f'{source}:{count}')})
                count+=1
                if count>=config.get('rows_per_source',32): break
    tasks=[]
    for index in range(90 if config.get('synthetic_tasks',True) else 0):
        a=index+2;b=index%11+3
        kind=index%3
        if kind==0:
            prompt=f'Calculate {a} * {b} + {index%7}. Return only the number.'
            answer=str(a*b+index%7);spec={'kind':'math','expected':answer}
        elif kind==1:
            prompt=f'Write one pure Python function solve(x) that returns x * {a} + {b}. Return only code.'
            answer=f'def solve(x):\n    return x * {a} + {b}'
            spec={'kind':'coding','verifier':'restricted_python','function':'solve',
                'cases':[{'args':[x],'expected':x*a+b} for x in [-3,0,7]]}
        else:
            prompt=f'Return a JSON object with exactly keys name and count. name is "item_{index}" and count is {a}.'
            spec={'kind':'structured','expected':{'name':f'item_{index}','count':a}}
            answer=json.dumps(spec['expected'])
        tasks.append({'prompt':prompt,'answer':answer,'task_spec':spec,'source':'local_verified_v1'})
    for path in config.get('verified_tasks',[]):
        with open(path) as stream:
            tasks.extend(json.loads(line) for line in stream if line.strip())
    plain=[task for task in tasks if 'messages' not in task]
    checked=dict(zip(map(id,plain),verify_many([task['answer'] for task in plain],[task['task_spec'] for task in plain],confirm=False)))
    for task in tasks:
        if 'messages' in task:
            # Agent transcript: one row per assistant turn, conditioned on exactly the context the model saw
            # at that turn (Gemma adds an empty thought block only to the live generation prompt).
            spec=json.loads(task['task_spec']) if isinstance(task['task_spec'],str) else task['task_spec']
            messages=task['messages']
            if messages[-1]['role']!='assistant' or not verify(messages[-1]['content'],spec).correct:
                raise ValueError('Transcript does not end in a verified answer')
            for turn,message in enumerate(messages):
                if message['role']!='assistant': continue
                context=tokenizer.apply_chat_template(messages[:turn],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                prefix=tokenizer.encode(context,add_special_tokens=False)
                suffix=tokenizer.encode(message['content'],add_special_tokens=False)+[tokenizer.eos_token_id]
                if len(prefix)+len(suffix)>config['sequence_length']: break  # never truncate a turn's context
                rows.append({'prompt':messages[0]['content'],'rendered':context,'input_ids':prefix+suffix,
                    'labels':[-100]*len(prefix)+suffix,'task_spec':spec,'source':task.get('source','agent_rft'),
                    'trace_id':task.get('trace_id',digest(messages))})
            continue
        prompt,answer,spec=task['prompt'],task['answer'],task['task_spec']
        if not isinstance(prompt,str) or not isinstance(answer,str): raise ValueError('Verified tasks require string prompt/answer')
        verdict=checked[id(task)]  # batched: Lean references share one Mathlib import per group
        if not verdict.verified or not verdict.correct: raise ValueError('Task reference failed its own verifier')
        rendered=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        prefix=tokenizer.encode(rendered,add_special_tokens=False)
        suffix=tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.eos_token_id]
        if len(prefix)+len(suffix)>config['sequence_length']: raise ValueError('Verified reference exceeds sequence budget')
        ids=prefix+suffix
        rows.append({'prompt':prompt,'rendered':rendered,'input_ids':ids,
            'labels':[-100]*len(prefix)+suffix,'task_spec':spec,'source':task.get('source','local_verified'),
            **({'trace_id':task['trace_id']} if 'trace_id' in task else {})})
    payload,audit=grouped_split(rows,config['seed'],config.get('near_duplicate_threshold',0.8))
    if audit['cross_split_pairs']: raise RuntimeError('Split leakage')
    return payload,audit


def deep_supervision(model):
    from gos_gemma import GoSReasoningLayer
    layer=inner_lm(model).get_output_embeddings()
    return isinstance(layer,GoSReasoningLayer) and torch.is_grad_enabled() and \
        layer.lora_A[layer.active_adapters[0]]._body.enhancements.get('deep_supervision',False)


def loss_for(model,row,variant,chunk=8):
    device=next(model.parameters()).device
    ids=torch.tensor([row['input_ids']],device=device);mask=torch.ones_like(ids)
    targets=torch.tensor(row['labels'][1:],device=device);valid=targets!=-100
    if not valid.any(): raise ValueError('No supervised continuation')
    lm=inner_lm(model)
    if variant in ('lora','recurrent'):
        hidden=lm.base_model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
    else:
        lm.base_model.eval()
        with torch.no_grad(): hidden=lm.base_model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
    hidden=hidden[0,:-1][valid];targets=targets[valid]
    if variant=='gos' and deep_supervision(model):
        from gos_gemma import deep_supervised_logps
        per_pass=deep_supervised_logps(model,hidden,targets,chunk)
        # Mean over passes of each pass's NLL; the final pass is the adapter's real output.
        return -per_pass.mean(),int(valid.sum()),len(row['input_ids'])
    if variant=='gos': logps,_=selected_logps(model,hidden,targets,chunk)
    else:
        head=lm.get_output_embeddings();cap=getattr(lm.config,'final_logit_softcapping',None)
        scale=getattr(lm.config,'logit_scale',None) or getattr(lm.config,'output_multiplier',None) or 1.0
        def project(x,y):
            logits=head(x)*scale
            if cap: logits=cap*torch.tanh(logits/cap)
            return torch.log_softmax(logits.float(),-1).gather(-1,y[:,None]).squeeze(-1)
        logps=torch.cat([checkpoint(project,hidden[i:i+chunk],targets[i:i+chunk],use_reentrant=False)
            if torch.is_grad_enabled() else project(hidden[i:i+chunk],targets[i:i+chunk]) for i in range(0,len(targets),chunk)])
    return -logps.mean(),int(valid.sum()),len(row['input_ids'])


def generate_measured(model,tokenizer,row,budget,cache):
    ids=tokenizer(row['rendered'],return_tensors='pt',add_special_tokens=False).to(model.device)
    if ids.input_ids.shape[1]+budget>4096: raise ValueError('Prompt + completion exceeds 4096-token cap')
    sync();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.no_grad(): output=model.generate(**ids,max_new_tokens=budget,do_sample=False,use_cache=cache,
        pad_token_id=tokenizer.pad_token_id,logits_to_keep=1)
    sync();elapsed=time.perf_counter()-start
    completion=output[0,ids.input_ids.shape[1]:];text=tokenizer.decode(completion,skip_special_tokens=True)
    verdict=verify(text,row['task_spec'])
    return {'text':text,'tokens':len(completion),'seconds':elapsed,'tokens_per_second':len(completion)/elapsed,
        'truncated':len(completion)==budget and int(completion[-1]) not in [tokenizer.eos_token_id],
        'peak_vram_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'verdict':verdict.as_dict(),'kind':row['task_spec']['kind'],'prompt_sha256':digest(row['prompt'])}


def profile_rollout(model,tokenizer,rows,config):
    """Exercise the actual TRL sequential rollout hook, including its scorer."""
    from types import SimpleNamespace
    from train_unified import sequential_rollout
    from rollout_cache import FrozenFeatureCache
    from gos_gemma import frozen_features
    profiles=[]
    for cached in (False,True):
        for feature_cached in (False,True):
            cache=FrozenFeatureCache(4)
            class Trainer:
                def _get_per_token_logps_and_entropies(self,model,ids,mask,completion_length):
                    features=cache.get(model,ids,mask,4096) if feature_cached else frozen_features(model,ids,mask)
                    features=features[:,-completion_length-1:-1].reshape(-1,features.shape[-1])
                    targets=ids[:,-completion_length:].reshape(-1)
                    logps,_=selected_logps(model,features,targets,config['logit_chunk'],temperature=1.0)
                    return logps.reshape(ids.shape[0],-1),None,None
            trainer=Trainer();trainer.model=model;trainer.processing_class=tokenizer;trainer.temperature=1.0
            trainer.gos_config=SimpleNamespace(grpo_use_cache=cached,max_prompt_tokens=config['sequence_length'],
                max_completion_tokens=config['profile_new_tokens'])
            trainer.accelerator=SimpleNamespace(unwrap_model=lambda x:x,device=model.device,autocast=contextlib.nullcontext)
            seed_all(config['seed']);sync();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            with torch.no_grad(): result=sequential_rollout([rows[0]['rendered']]*(2*config.get('profile_repeats',1)),trainer)
            sync();seconds=time.perf_counter()-start
            profiles.append({'native_cache':cached,'frozen_feature_cache':feature_cached,'seconds':seconds,
                'tokens':sum(map(len,result['completion_ids'])),'peak_vram_bytes':torch.cuda.max_memory_allocated(),
                'feature_hits':cache.hits,'feature_misses':cache.misses,'result_sha256':digest(result),
                'tokens_per_second':sum(map(len,result['completion_ids']))/seconds})
    return profiles


def heldout_evaluation(model,tokenizer,config,train,output):
    """Sampled pass@k / maj@k on held-out task files that never enter training or the split."""
    from best_of_n import evaluate_row,hf_generator
    from build_eval_set import jaccard_flags
    from eval_search import read_jsonl,summarize
    rows=[row for path in config['heldout_tasks'] for row in read_jsonl(path)]
    caps=config.get('heldout_per_source',{})  # deterministic file-order caps; never chosen by outcomes
    counts={}
    def keep(row):
        counts[row['source']]=counts.get(row['source'],0)+1
        return counts[row['source']]<=caps.get(row['source'],len(rows))
    rows=[row for row in rows if keep(row)][:config.get('heldout_limit')]
    probe=[{'prompt':row['prompt']} for row in rows]
    jaccard_flags(probe,[r['prompt'] for r in train],config.get('heldout_leak_threshold',0.5))
    leaks=[row['id'] for row,item in zip(rows,probe) if item['contamination']['flagged'] and item['contamination']['match']=='reference']
    if leaks: raise RuntimeError(f'Held-out rows overlap training rows: {leaks[:5]}')
    k=config.get('heldout_samples',4);ks=[x for x in (1,2,4,8,16) if x<=k]
    seed_all(config['seed']);model.eval()
    generate=hf_generator(model,tokenizer,config.get('heldout_max_new_tokens',1024),config.get('heldout_temperature',0.7),
        config.get('heldout_batch',4),thinking=config.get('heldout_thinking',False),context=config.get('heldout_context',4096))
    records=[evaluate_row(row,generate,k,config.get('heldout_repair',True)) for row in rows]
    Path(output,'heldout_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    return summarize(records,ks)


def evaluate_split(model,tokenizer,config,variant,validation,evaluation,train,output):
    Path(output).mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        val=[float(loss_for(model,row,variant,config['logit_chunk'])[0]) for row in validation[:config.get('validation_rows',2)]]
    results={str(cache):[generate_measured(model,tokenizer,r,config['max_new_tokens'],cache) for r in evaluation] for cache in (False,True)}
    heldout=heldout_evaluation(model,tokenizer,config,train,output) if config.get('heldout_tasks') else None
    accuracy={cache:sum(x['verdict']['correct'] for x in items)/len(items) for cache,items in results.items()}
    return {'validation_nll':val,'test_outputs':results,'test_accuracy':accuracy,'heldout':heldout}


def worker(config,variant,depth,output,resume=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seed_all(config['seed']);model,tokenizer=load_local_gemma(config['base_path'])
    enhancements={'scratchpad':True,'latent_slots':2,'thermal':True,'dual_friction':True,
        'edge_routing':True,'auxiliary_critics':True,'min_rounds':min(2,depth)}|config.get('enhancements',{})
    from train_unified import validate_enhancements
    validate_enhancements(enhancements,depth)
    model=configure(model,variant,config['width'],depth,enhancements,config.get('lora_rank',2),config.get('lora_targets','qv'),config.get('recurrent'))
    rows,audit=make_dataset(config,tokenizer);write(output/'split_audit.json',audit)
    write(output/'dataset_manifest.json',rows)
    train=[r for r in rows if r['split']=='train' and any(x!=-100 for x in r['labels'][1:])]
    validation=[r for r in rows if r['split']=='validation']
    evaluation=[r for r in rows if r['split']=='test' and 'task_spec' in r and r['source']!='agent_rft']
    # Balanced task coverage where present. No selection using model outcomes.
    evaluation=[r for kind in ('math','coding','structured') for r in [x for x in evaluation if x['task_spec']['kind']==kind][:config['eval_per_kind']]]
    if not train or not evaluation: raise ValueError('Empty training or test partition')
    rng=random.Random(config['seed']);rng.shuffle(train)
    # Interleave available sources so a short pilot actually trains on local distillation rows.
    pools={source:[r for r in train if r['source']==source] for source in sorted({r['source'] for r in train})}
    train=[]
    while any(pools.values()):
        for pool in pools.values():
            if pool: train.append(pool.pop())
    metadata={'config_sha256':digest(config),'data_sha256':audit['sha256'],'variant':variant,'depth':depth}
    params=[p for p in model.parameters() if p.requires_grad]
    lr=config.get('lora_learning_rate',config['learning_rate']) if variant in ('lora','recurrent') else config['learning_rate']
    optimizer=torch.optim.AdamW(params,lr=lr) if params else None
    start_step=load_checkpoint(output/'checkpoint.pt',model,optimizer,metadata) if resume else 0
    if not resume and (output/'checkpoint.pt').exists(): raise FileExistsError('Use --resume or a fresh output directory')
    before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    history=[];sync();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();tokens=supervised=0
    for step in range(start_step,config['steps'] if params else 0):
        model.train();optimizer.zero_grad(set_to_none=True)
        if variant=='recurrent':
            # Random pass count per step, keyed on (seed, step) so resume replays the same schedule.
            from gos_recurrent import sample_passes,set_passes
            spec=config.get('recurrent',{})
            set_passes(model,sample_passes(config['seed'],step,spec),spec.get('backprop_passes'))
        loss,n,total=loss_for(model,train[step%len(train)],variant,config['logit_chunk'])
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step()
        tokens+=total;supervised+=n;history.append(float(loss.detach()))
        save_checkpoint(output/'checkpoint.pt',model,optimizer,step+1,metadata)
        print(json.dumps({'variant':variant,'depth':depth,'step':step+1,'loss':history[-1]}),flush=True)
    sync();seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    model.eval()
    by_passes={}
    if variant=='recurrent':
        from gos_recurrent import set_passes
        for passes in config.get('recurrent',{}).get('eval_passes',[1,config.get('recurrent',{}).get('max_passes',4)]):
            set_passes(model,passes)
            by_passes[str(passes)]=evaluate_split(model,tokenizer,config,variant,validation,evaluation,train,output/f'passes-{passes}')
        model_eval=by_passes[str(passes)]  # headline numbers: the last (largest) pass count
    else:
        model_eval=evaluate_split(model,tokenizer,config,variant,validation,evaluation,train,output)
    val,results=model_eval['validation_nll'],model_eval['test_outputs']
    metrics={}
    for cache,items in results.items():
        duration=sum(x['seconds'] for x in items);generated=sum(x['tokens'] for x in items)
        metrics[cache]={'accuracy':sum(x['verdict']['correct'] for x in items)/len(items),
            'verified_coverage':sum(x['verdict']['verified'] for x in items)/len(items),
            'samples':len(items),'peak_vram_bytes':max(x['peak_vram_bytes'] for x in items),'tokens_per_second':generated/duration,'mean_latency_seconds':duration/len(items),
            'truncations':sum(x['truncated'] for x in items),
            'by_kind':{k:{'correct':sum(x['verdict']['correct'] for x in items if x['kind']==k),
                'total':sum(x['kind']==k for x in items)} for k in ('math','coding','structured')}}
    agreement=sum(a['text']==b['text'] for a,b in zip(results['False'],results['True']))/len(evaluation)
    report={'format_version':1,'metadata':metadata,'config':config,'enhancements':enhancements,
        'trainable_parameters':trainable_count(model),'training':{'steps_completed':start_step+len(history),
        'steps_this_run':len(history),'changed_parameter_tensors':sum(not torch.equal(before[n],p.detach().cpu()) for n,p in model.named_parameters() if p.requires_grad),
        'parameter_update_l2':sum(float((p.detach().cpu()-before[n]).double().square().sum()) for n,p in model.named_parameters() if p.requires_grad)**0.5,'seconds':seconds,'tokens_per_second':tokens/seconds,
        'supervised_tokens_per_second':supervised/seconds,'steps_per_second':len(history)/seconds,
        'losses':history,'consumed_sources':{source:sum(train[step%len(train)]['source']==source for step in range(start_step,start_step+len(history))) for source in sorted({r['source'] for r in train})},'training_sources':{source:sum(r['source']==source for r in train) for source in sorted({r['source'] for r in train})},'peak_vram_bytes':peak,'peak_reserved_bytes':reserved},
        'validation_nll':val,'test_metrics':metrics,'test_outputs':results,'cache_text_agreement':agreement,
        'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'determinism':'seeded; deterministic algorithms warn_only; TF32 disabled','versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','peft','trl','bitsandbytes']},
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        'limitations':['Short pilot; no reasoning improvement established.','Lexical near-duplicate audit is not semantic equivalence.',
            'Restricted Python tasks cover a safe pure-function subset.','Deterministic seeds do not guarantee bitwise CUDA reproducibility.']}
    if by_passes:
        report['by_passes']={p:{k:v for k,v in item.items() if k!='test_outputs'} for p,item in by_passes.items()}
    if model_eval.get('heldout') is not None:
        report['heldout']=model_eval['heldout']
    if variant=='gos':
        if config.get('measure_eviction_quality',False):
            from cache_quality import cache_quality
            ids=torch.tensor([evaluation[0]['input_ids'][:config.get('cache_quality_tokens',24)]],device=model.device)
            report['inference_cache_quality']=[cache_quality(model,ids,heavy,recent,quant) for heavy,recent,quant in [(0,4096,'none'),(4,4,'none'),(0,8,'none'),(4,4,'int4')]]
        report['grpo_rollout_profile']=profile_rollout(model,tokenizer,evaluation,config)
        export_gos_gguf(model,output/'gos-adapter.gguf',{'width':config['width'],'rounds':depth,
            'enhancements':enhancements,'base_path':config['base_path'],'experiment':metadata})
        from gos_gemma import save_gos_adapter
        save_gos_adapter(model,output/'adapter',{'graph_nodes':4,'graph_rounds':depth,'graph_enhancements':enhancements})
        # Verify the exported trained head against the same immutable frozen features.
        from experiment_support import load_gos_gguf
        from gos_gemma import frozen_features
        probe=torch.tensor([evaluation[0]['input_ids'][:8]],device=model.device)
        with torch.no_grad():
            hidden=frozen_features(model,probe,torch.ones_like(probe))[:,-1]
            expected=inner_lm(model).get_output_embeddings()(hidden).float()
            base=model.unload();restored,_=load_gos_gguf(base,output/'gos-adapter.gguf')
            actual=inner_lm(restored).get_output_embeddings()(hidden).float()
            error=float((actual-expected).abs().max())
        if error!=0: raise RuntimeError('GGUF head roundtrip changed logits')
        report['gguf_head_roundtrip_max_logit_error']=error
    elif variant in ('lora','recurrent'): model.save_pretrained(output/'adapter',save_embedding_layers=False)
    if variant=='recurrent':
        from gos_recurrent import save_recurrence
        save_recurrence(model,output/'adapter')
    write(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--worker',choices=['frozen','lora','mlp','gos','recurrent']);parser.add_argument('--depth',type=int,default=2)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    required=['format_version','base_path','seed','steps','sequence_length','width','learning_rate','eval_per_kind','max_new_tokens','logit_chunk','profile_new_tokens']
    if any(k not in config for k in required) or config['format_version']!=1: raise ValueError('Invalid versioned configuration')
    if not 8<=config['sequence_length']<=4096 or config['steps']<1 or config['width']<1 or args.depth<1 or config['logit_chunk']<1 or config['learning_rate']<=0 or config['eval_per_kind']<1 or config['max_new_tokens']<1 or config['profile_new_tokens']<1 or any(d<1 for d in config.get('depths',[])): raise ValueError('Invalid training budget')
    output=Path(args.output)
    if args.worker: worker(config,args.worker,args.depth,output,args.resume);return
    output.mkdir(parents=True,exist_ok=True);write(output/'configuration.json',config)
    base=config.get('base_variants',['frozen','lora','mlp','gos'])  # which standard arms run before extra_variants
    if not set(base)<={'frozen','lora','mlp','gos'}: raise ValueError('Unknown base variant')
    variants=[(v,config['depths'][0]) for v in base if v!='gos']+([('gos',d) for d in config['depths']] if 'gos' in base else [])
    reports=[]
    for variant,depth in variants:
        directory=output/f'{variant}-d{depth}'
        command=[sys.executable,__file__,'--config',args.config,'--output',str(directory),'--worker',variant,'--depth',str(depth)]
        if args.resume and variant!='frozen' and (directory/'checkpoint.pt').exists(): command.append('--resume')
        if not (args.resume and (directory/'report.json').exists()):
            with (output/f'{variant}-d{depth}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        reports.append(json.loads((directory/'report.json').read_text()))
    for ablation in config.get('ablations',[]):
        switches={'recurrent_memory':{'latent_slots':0},'holographic_scratchpad':{'scratchpad':False},
            'edge_routing':{'edge_routing':False},'thermal_recursion':{'thermal':False},'auxiliary_critics':{'auxiliary_critics':False}}
        if ablation not in switches: raise ValueError('Unknown ablation')
        changed=dict(config);changed['enhancements']=config.get('enhancements',{})|switches[ablation];changed['ablations']=[]
        filename=output/f'config-without-{ablation}.json';write(filename,changed)
        directory=output/f'gos-without-{ablation}'
        with (output/f'gos-without-{ablation}.log').open('w') as log:
            subprocess.run([sys.executable,__file__,'--config',str(filename),'--output',str(directory),
                '--worker','gos','--depth',str(config['depths'][-1])],stdout=log,stderr=subprocess.STDOUT,check=True)
        reports.append(json.loads((directory/'report.json').read_text()))
    for extra in config.get('extra_variants',[]):
        # {'name', 'worker', 'depth'?, 'enhancements'?, 'overrides'?}: extra arms with their own configs.
        changed=dict(config)|extra.get('overrides',{});changed['ablations']=[];changed['extra_variants']=[]
        changed['enhancements']=config.get('enhancements',{})|extra.get('enhancements',{})
        filename=output/f"config-{extra['name']}.json";write(filename,changed)
        directory=output/extra['name']
        command=[sys.executable,__file__,'--config',str(filename),'--output',str(directory),
            '--worker',extra.get('worker','gos'),'--depth',str(extra.get('depth',config['depths'][-1]))]
        if args.resume and (directory/'checkpoint.pt').exists(): command.append('--resume')
        if not (directory/'report.json').exists():
            with (output/f"{extra['name']}.log").open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        reports.append(json.loads((directory/'report.json').read_text()))
    write(output/'comparison.json',{'format_version':1,'configuration_sha256':digest(config),'experiments':reports})

if __name__=='__main__':main()
