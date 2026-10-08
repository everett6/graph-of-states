"""PEFT-managed GoS residual head for a completely frozen causal language model."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer

from gos_engine_architecture import EvidenceGraphTransition, LRRConfig
from gos_cognition import CrossModalCoherence, AsynchronousMatrixBranches
from gos_memory import HolographicScratchpad, RecurrentLatentCompactor, thermal_statistics


class _TokenGraphBody(nn.Module):
    """Reconstruct each causal token's latent state without seeing future tokens."""
    def __init__(self, hidden_size, width, nodes=4, rounds=4, enhancements=None):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(hidden_size, width), nn.LayerNorm(width))
        self.transition = EvidenceGraphTransition(LRRConfig(width=width, nodes=nodes))
        self.fuse = nn.Linear(2 * width, width)
        self.rounds = rounds
        self.enhancements = enhancements or {}
        options = self.enhancements
        self.scratchpad = HolographicScratchpad(width, nodes) if options.get('scratchpad', False) else None
        slots = options.get('latent_slots', 0)
        self.compactor = RecurrentLatentCompactor(width, slots) if slots else None
        if self.scratchpad is not None or self.compactor is not None:
            self.memory_gate = nn.Parameter(torch.tensor(-2.0))
        self.coherence = CrossModalCoherence(width) if options.get('cross_modal', False) else None
        self.branches = AsynchronousMatrixBranches(width, nodes, options.get('async_branches', False)) if options.get('branches', False) else None
        self.operational = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1)) if options.get('dual_friction', False) else None
        self.actuator = nn.Linear(width, 1) if options.get('tool_gate', False) else None

    def forward(self, hidden, diagnostics=False, modal_states=None, modal_mask=None, allow_async=False):
        # TRL may cast adapter parameters to BF16. Direct head scoring bypasses
        # the causal LM wrapper, so also work correctly without wrapper autocast.
        anchor = self.project(hidden.to(self.project[0].weight.dtype))
        coherence_loss = anchor.sum() * 0
        if modal_states is not None:
            if self.coherence is None:
                raise ValueError('Enable cross_modal to supply modality evidence')
            if modal_states.ndim != 3 or modal_states.shape[0] != len(hidden) or modal_states.shape[-1] != hidden.shape[-1] or modal_states.shape[1] < 1:
                raise ValueError('modal_states must be [tokens, modalities, backbone_hidden_width]')
            if modal_mask is None or modal_mask.shape != modal_states.shape[:2]:
                raise ValueError('Provide a causal aligned modal_mask')
            modalities = self.project(modal_states.to(anchor.dtype))
            anchor, coherence_loss = self.coherence(anchor, modalities, modal_mask.bool())
        state = anchor[:, None] + self.transition.roles[None]
        options = self.enhancements
        scratch = torch.zeros_like(anchor)
        slots = options.get('latent_slots', 0)
        memory = anchor[:, None].expand(-1, slots, -1).clone()
        temperature = anchor.new_full((len(anchor),), options.get('temperature', 1.0))
        entropy = anchor.new_ones(len(anchor))
        active = torch.ones(len(anchor), dtype=torch.bool, device=anchor.device)
        steps = torch.zeros(len(anchor), dtype=torch.long, device=anchor.device)
        risk = state.new_zeros(state.shape[:2])
        semantic_risk = operational_risk = risk
        for index in range(self.rounds):
            previous = state
            evidence = anchor[:, None]
            if self.scratchpad is not None:
                evidence = torch.cat((evidence, self.scratchpad.read(scratch) * self.memory_gate.sigmoid()), 1)
            if self.compactor is not None:
                evidence = torch.cat((evidence, memory * self.memory_gate.sigmoid()), 1)
            mask = torch.ones(evidence.shape[:2], dtype=torch.bool, device=anchor.device)
            branch_state = state
            if self.branches is not None:
                branch_state = state + 0.1 * self.branches(state, allow_async=allow_async)
            operational = self.operational(torch.cat((branch_state, anchor[:, None].expand_as(state)), -1)).squeeze(-1) if self.operational is not None else None
            proposal, proposed_risk, routes, semantic = self.transition(branch_state, evidence, mask,
                temperature=temperature[:, None, None] if options.get('thermal', False) else 1.0,
                operational_logits=operational, return_components=True)
            semantic_risk = torch.where(active[:, None], semantic, semantic_risk)
            if operational is not None:
                operational_risk = torch.where(active[:, None], operational, operational_risk)
            state = torch.where(active[:, None, None], proposal, previous)
            risk = torch.where(active[:, None], proposed_risk, risk)
            steps = steps + active.long()
            if self.scratchpad is not None:
                scratch = torch.where(active[:, None], self.scratchpad.write(scratch, state), scratch)
            if self.compactor is not None:
                memory = torch.where(active[:, None, None], self.compactor(memory, state), memory)
            next_entropy, change, motion, temperature = thermal_statistics(
                routes, previous, state, entropy, options.get('temperature', 1.0),
                options.get('temperature_floor', 0.5))
            temperature = temperature.to(anchor.dtype)
            entropy = torch.where(active, next_entropy.to(entropy.dtype), entropy)
            if options.get('thermal', False) and index + 1 >= options.get('min_rounds', 2):
                converged = (change <= options.get('entropy_tolerance', 0.01)) & (motion <= options.get('motion_tolerance', 0.01))
                active = active & ~converged
                # Same decision during training/scoring/generation. Threshold is
                # discrete; no gradient is claimed through the stopping decision.
                if not bool(active.any()):
                    break
        features = torch.tanh(self.fuse(torch.cat((state.mean(1), state.var(1, unbiased=False)), -1)))
        if diagnostics:
            return features, risk.mean(-1), {'rounds': steps, 'entropy': entropy,
                'temperature': temperature, 'latent_slots': memory.shape[1],
                'semantic_risk': semantic_risk.mean(-1), 'operational_risk': operational_risk.mean(-1),
                'coherence_loss': coherence_loss,
                'tool_logits': self.actuator(features).squeeze(-1) if self.actuator is not None else None}
        return features, risk.mean(-1)

class TokenGraph(nn.Module):
    """Flat adapter parameter names preserve PEFT save/load conventions.

    PEFT expects adapter names immediately before each parameter name. A
    stateless functional body lets a recursive graph obey that convention
    without a custom checkpoint format or global patches to PEFT.
    """
    def __init__(self, hidden_size, width, nodes=4, rounds=4, enhancements=None):
        super().__init__()
        body = _TokenGraphBody(hidden_size, width, nodes, rounds, enhancements)
        self.parameter_names = []
        for index, (name, parameter) in enumerate(body.named_parameters()):
            self.register_parameter(f'p{index}', parameter)
            self.parameter_names.append(name)
        object.__setattr__(self, '_body', body)

    def features_and_risk(self, hidden, diagnostics=False, modal_states=None, modal_mask=None):
        parameters = {name: getattr(self, f'p{index}') for index, name in enumerate(self.parameter_names)}
        kwargs = {'diagnostics': diagnostics, 'modal_states': modal_states, 'modal_mask': modal_mask,
                  'allow_async': not self.training and not torch.is_grad_enabled()}
        # Normal .to() and state-dict copies preserve Parameter identity. Avoid
        # a stateless reparameterization context in that common case. PEFT or
        # assign=True replacements still take the safe functional fallback.
        template = dict(self._body.named_parameters())
        if all(template[name] is parameter for name, parameter in parameters.items()):
            return self._body(hidden, **kwargs)
        return torch.func.functional_call(self._body, parameters, (hidden,), kwargs)

    def forward(self, hidden, modal_states=None, modal_mask=None):
        return self.features_and_risk(hidden, modal_states=modal_states, modal_mask=modal_mask)[0]


class GoSReasoningLayer(nn.Module, LoraLayer):
    """Custom PEFT adapter: hidden → graph → residual hidden → frozen LM head.

    lora_A/B are PEFT's adapter containers, not standard low-rank linear LoRA.
    The adapter is nonlinear and cannot be merged into the frozen LM matrix.
    """
    def __init__(self, base_layer, adapter_name, r=128, lora_alpha=128, config=None, **kwargs):
        nn.Module.__init__(self)
        LoraLayer.__init__(self, base_layer)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, config=config)

    def update_layer(self, adapter_name, r, lora_alpha=128, config=None, **kwargs):
        graph = getattr(config, '_gos_config', {'nodes': 4, 'rounds': 4})
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = 1.0
        self.lora_dropout[adapter_name] = nn.Identity()
        self.lora_A[adapter_name] = TokenGraph(self.in_features, r, graph['nodes'], graph['rounds'], graph.get('enhancements'))
        self.lora_B[adapter_name] = nn.Linear(r, self.in_features, bias=False)
        nn.init.zeros_(self.lora_B[adapter_name].weight)
        self.lora_A[adapter_name].to(device=self.base_layer.weight.device, dtype=torch.float32)
        self.lora_B[adapter_name].to(device=self.base_layer.weight.device, dtype=torch.float32)
        self.use_dora[adapter_name] = False
        self.use_rslora[adapter_name] = False
        self.lora_bias[adapter_name] = False

    def forward(self, hidden, *args, modal_states=None, modal_mask=None, **kwargs):
        shape = hidden.shape
        # Backbone never participates in autograd, even for generation scoring.
        flat = hidden.detach().reshape(-1, shape[-1])
        reconstructed = flat
        if not self.disable_adapters:
            for name in self.active_adapters:
                graph = self.lora_A[name]
                delta = self.lora_B[name](graph(flat, modal_states, modal_mask))
                reconstructed = reconstructed + delta.to(flat.dtype)
        return self.base_layer(reconstructed.reshape(shape), *args, **kwargs)

    def risk_logits(self, hidden, stream='combined'):
        if stream not in ('combined', 'semantic', 'operational'):
            raise ValueError('Unknown friction stream')
        graph = self.lora_A[self.active_adapters[0]]
        features, combined, diagnostics = graph.features_and_risk(hidden.detach().reshape(-1, hidden.shape[-1]), diagnostics=True)
        if stream == 'combined' or not graph._body.enhancements.get('dual_friction', False):
            return combined
        return diagnostics[stream + '_risk']

    def tool_logits(self, hidden):
        graph = self.lora_A[self.active_adapters[0]]
        diagnostics = graph.features_and_risk(hidden.detach().reshape(-1, hidden.shape[-1]), diagnostics=True)[2]
        if diagnostics['tool_logits'] is None:
            raise ValueError('Enable tool_gate before training or using an actuator')
        return diagnostics['tool_logits']

    def merge(self, *args, **kwargs):
        raise RuntimeError('A nonlinear GoS adapter cannot be merged; load it with load_gos_adapter')

    def unmerge(self):
        raise RuntimeError('GoS adapters are never merged')


def gos_peft_config(width=128, nodes=4, rounds=4, enhancements=None):
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=['lm_head'],
                        r=width, lora_alpha=width, lora_dropout=0.0,
                        bias='none', ensure_weight_tying=False)
    config._gos_config = {'nodes': nodes, 'rounds': rounds, 'enhancements': enhancements or {}}
    config._register_custom_module({nn.Linear: GoSReasoningLayer})
    return config


def attach_gos(base, width=128, nodes=4, rounds=4, enhancements=None):
    # PEFT performs freezing and adapter lifecycle management without k-bit
    # preparation's FP32 upcast of the entire embedding/LM-head matrices.
    base.requires_grad_(False)
    model = get_peft_model(base, gos_peft_config(width, nodes, rounds, enhancements))
    assert_gos_only_trainable(model)
    return model


def assert_gos_only_trainable(model):
    allowed = {id(parameter) for module in model.modules() if isinstance(module, GoSReasoningLayer)
               for name, parameter in module.named_parameters() if not name.startswith('base_layer.')}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(id(parameter) not in allowed for parameter in trainable):
        raise RuntimeError('Only parameters inside GoSReasoningLayer may be trainable')
    for module in model.modules():
        if isinstance(module, GoSReasoningLayer) and any(p.requires_grad for p in module.base_layer.parameters()):
            raise RuntimeError('Frozen LM head has trainable parameters')
    return sum(p.numel() for p in trainable)


def save_gos_adapter(model, path, graph_config):
    path = Path(path)
    model.save_pretrained(path, save_embedding_layers=False)
    (path / 'gos_config.json').write_text(json.dumps(graph_config, indent=2) + '\n')


def load_gos_adapter(base, path, trainable=False):
    graph = json.loads((Path(path) / 'gos_config.json').read_text())
    config = LoraConfig.from_pretrained(path)
    config._gos_config = {'nodes': graph['graph_nodes'], 'rounds': graph['graph_rounds'],
                          'enhancements': graph.get('graph_enhancements', {})}
    config._register_custom_module({nn.Linear: GoSReasoningLayer})
    model = PeftModel.from_pretrained(base, path, config=config, is_trainable=trainable)
    if trainable:
        assert_gos_only_trainable(model)
    return model


def inner_lm(model):
    return model.get_base_model() if isinstance(model, PeftModel) else model


def frozen_features(model, input_ids, attention_mask, max_seq_length=4096):
    if input_ids.shape[1] > max_seq_length or max_seq_length > 4096:
        raise ValueError('Total sequence length exceeds the 4096-token hard cap')
    lm = inner_lm(model)
    # Supported causal LM architectures expose their decoder via base_model.
    backbone = lm.base_model
    backbone.eval()
    with torch.no_grad():
        return backbone(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, return_dict=True, output_hidden_states=False,
                        output_attentions=False).last_hidden_state.detach()


def selected_logps(model, hidden, targets, chunk_size=32, temperature=1.0, entropy=False,
                   modal_states=None, modal_mask=None, checkpoint_backend="torch"):
    """Checkpoint each small vocabulary projection; no [4096, vocab] tensor."""
    lm = inner_lm(model)
    head = lm.get_output_embeddings()
    config = lm.config.get_text_config()
    softcap = getattr(config, 'final_logit_softcapping', None)
    scale = getattr(config, 'logit_scale', None)
    if scale is None:
        scale = getattr(config, 'output_multiplier', 1.0)
    scale = 1.0 if scale is None else scale

    def project(features, labels, modalities=None, modalities_mask=None):
        logits = head(features, modal_states=modalities, modal_mask=modalities_mask) * scale
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        probabilities = F.log_softmax(logits.float() / temperature, -1)
        chosen = probabilities.gather(-1, labels[:, None]).squeeze(-1)
        entropies = -(probabilities.exp() * probabilities).sum(-1) if entropy else chosen * 0
        return chosen, entropies

    logps, entropies = [], []
    for start in range(0, len(targets), chunk_size):
        arguments = (hidden[start:start+chunk_size], targets[start:start+chunk_size],
            modal_states[start:start+chunk_size] if modal_states is not None else None,
            modal_mask[start:start+chunk_size] if modal_mask is not None else None)
        if torch.is_grad_enabled():
            from gos_runtime import checkpoint_call
            values = checkpoint_call(project, arguments, checkpoint_backend)
        else:
            values = project(*arguments)
        logps.append(values[0])
        entropies.append(values[1])
    return torch.cat(logps), torch.cat(entropies)
