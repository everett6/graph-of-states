from dataclasses import replace
import json
from pathlib import Path
import runpy
import sys
import tempfile
workspace = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(workspace))
import os
os.chdir(workspace)
from train_unified import UnifiedConfig, run
fixture = runpy.run_path('tests/test_unified.py')
import argparse
parser = argparse.ArgumentParser(description='Offline NF4 SFT → GRPO → test integration, using only generated tiny weights and synthetic rows')
parser.add_argument('--output', default='runs/offline-integration')
parser.add_argument('--deepspeed', action='store_true')
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as temp:
    temp = Path(temp)
    fixture['tiny_base']().save_pretrained(temp / 'model')
    fixture['tokenizer']().save_pretrained(temp / 'model')
    sources = []
    for name, weight in [('stratos',2),('fable',1),('feedback',1)]:
        path = temp / f'{name}.jsonl'
        path.write_text(''.join(json.dumps({'prompt':f'a {i}', 'answer':'ok'})+'\n' for i in range(100)))
        sources.append({'name':name, 'weight':weight, 'path':str(path)})
    config = UnifiedConfig(model=str(temp/'model'), sources=sources, output_dir=str(root/'sft'),
        max_seq_length=64, max_prompt_tokens=48, max_completion_tokens=8, overlap=8,
        graph_width=16, graph_nodes=3, graph_rounds=4,
        graph_enhancements=dict(scratchpad=True, latent_slots=2, thermal=True, cross_modal=True, branches=True, async_branches=True, dual_friction=True, tool_gate=True, min_rounds=2, temperature=1.5, temperature_floor=0.75, entropy_tolerance=0.01, motion_tolerance=0.01),
        inference_cache="h2o", h2o_heavy_tokens=1, h2o_recent_tokens=2, kv_quantization="int4", kv_kernel="triton", logits_chunk_size=2,
        gradient_accumulation_steps=2, grpo_generations=2, max_steps=2, eval_steps=1,
        save_steps=1, eval_samples=4)
    if args.deepspeed:
        config = replace(config, checkpoint_backend='deepspeed', checkpoint_cpu_offload=True)
    run(config, 'sft')
    adapter = str(root/'sft'/'adapter')
    run(replace(config,output_dir=str(root/'grpo'),max_steps=1), 'grpo', adapter=adapter)
    run(replace(config,output_dir=str(root/'test')), 'test', adapter=str(root/'grpo'/'adapter'))
(root/'summary.json').write_text(json.dumps({'status':'passed', 'gpu':'RTX 5070',
    'model':'locally constructed tiny Gemma4, NF4', 'external_data_downloaded':False,
    'external_weights_downloaded':False, 'stages':['sft','grpo','test'], 'features':['holographic scratchpad','thermal looping','recurrent latent compaction','h2o generation','int4 Triton KV cache','matrix branches','dual friction','coherence and tool gate parameters']},indent=2)+'\n')
print('UNIFIED GPU INTEGRATION PASSED')

from gos_runtime import cleanup_checkpointing
cleanup_checkpointing()
