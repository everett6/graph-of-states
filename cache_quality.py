"""Teacher-forced cache distortion measurement; no claims from greedy agreement alone."""
import torch
from gos_h2o import H2OConfig,h2o_session


@torch.no_grad()
def cache_quality(model,ids,heavy=4,recent=4,quantization='none'):
    model.eval()
    reference=model(input_ids=ids,use_cache=False).logits.float()
    native=[];past=None
    for position in range(ids.shape[1]):
        out=model(input_ids=ids[:,position:position+1],past_key_values=past,use_cache=True,logits_to_keep=1)
        past=out.past_key_values;native.append(out.logits.float())
    native=torch.cat(native,1);del past
    values=[]
    with h2o_session(model,H2OConfig(heavy,recent,quantization)) as session:
        for position in range(ids.shape[1]):
            session['position']=position
            out=model(input_ids=ids[:,position:position+1],position_ids=torch.tensor([[position]],device=ids.device),
                attention_mask={'full_attention':None,'sliding_attention':None},use_cache=False,logits_to_keep=1)
            values.append(out.logits.float())
        evicted=sum(cache.evicted for cache in session['layers'].values())
        storage=sum(cache.storage_bytes() for cache in session['layers'].values())
    actual=torch.cat(values,1)
    logp=reference.log_softmax(-1);logq=actual.log_softmax(-1)
    return {'tokens':ids.shape[1],'heavy':heavy,'recent':recent,'quantization':quantization,
        'max_logit_error':float((actual-reference).abs().max()),
        'native_incremental_max_error_to_full':float((native-reference).abs().max()),
        'max_error_to_native_incremental':float((actual-native).abs().max()),
        'mean_kl_native_to_cache':float((native.log_softmax(-1).exp()*(native.log_softmax(-1)-actual.log_softmax(-1))).sum(-1).mean()),
        'mean_kl_reference_to_cache':float((logp.exp()*(logp-logq)).sum(-1).mean()),
        'top1_agreement':float((actual.argmax(-1)==reference.argmax(-1)).float().mean()),
        'evictions':evicted,'resident_kv_bytes':storage}
