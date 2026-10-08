"""Latent Recursive Reconstruction: an experimental, bounded graph reasoner.

Run `python gos_engine_architecture.py --smoke-test` for a CPU training check.
Only PyTorch is required for the core; Hugging Face loading is optional.
"""
from dataclasses import dataclass, asdict
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class LRRConfig:
    backbone_dim: int = 128
    vocab_size: int = 256
    width: int = 128
    nodes: int = 4
    rounds: int = 4
    decoder_layers: int = 1
    min_rounds: int = 2
    halt_tolerance: float = 0.005
    checkpoint_rounds: bool = False

    def __post_init__(self):
        if min(self.backbone_dim, self.vocab_size, self.width, self.nodes,
               self.rounds, self.decoder_layers, self.min_rounds) < 1:
            raise ValueError('Dimensions and iteration counts must be positive')
        if self.min_rounds > self.rounds or self.halt_tolerance <= 0:
            raise ValueError('Invalid halting configuration')


class EvidenceGraphTransition(nn.Module):
    """Shared transition with evidence anchoring and directed compatibility edges.

    Candidate nodes query prompt evidence independently. Directed edges compare
    displacement from each candidate's parent, not just candidate similarity.
    Incoming messages favor compatible, low-risk candidates. A gated residual
    update preserves the destination hypothesis; disagreement is retained for
    the decoder rather than erased by a single weighted average.
    """
    def __init__(self, c):
        super().__init__()
        d = c.width
        self.roles = nn.Parameter(torch.randn(c.nodes, d) / math.sqrt(d))
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.propose = nn.Sequential(nn.Linear(3*d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.edge = nn.Sequential(nn.Linear(3*d, d), nn.GELU(), nn.Linear(d, 1))
        self.risk = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.risk[-1].weight)
        nn.init.zeros_(self.risk[-1].bias)
        self.gate = nn.Linear(3*d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, state, evidence, mask, temperature=1.0, operational_logits=None, return_components=False):
        d = state.shape[-1]
        attention = self.query(state) @ self.key(evidence).transpose(-1, -2) / math.sqrt(d)
        attention = attention / temperature
        attention = attention.masked_fill(~mask[:, None, :], float('-inf')).softmax(-1)
        anchor = attention @ evidence
        role = self.roles[None].expand(state.shape[0], -1, -1)
        displacement = torch.tanh(self.propose(torch.cat((state, anchor, role), -1)))
        candidate = self.norm(state + displacement)
        n = state.shape[1]
        # Axis 1 is destination; axis 2 is source.
        destination = candidate[:, :, None, :].expand(-1, -1, n, -1)
        source = candidate[:, None, :, :].expand(-1, n, -1, -1)
        delta_difference = displacement[:, :, None, :] - displacement[:, None, :, :]
        compatibility = self.edge(torch.cat((destination, source, delta_difference), -1)).squeeze(-1)
        risk_logits = self.risk(torch.cat((candidate, anchor), -1)).squeeze(-1)
        semantic_logits = risk_logits
        if operational_logits is not None:
            # P(any failure) = 1 - P(no semantic failure)*P(no operational failure).
            # This factorization is a modeling choice, not a correctness proof.
            log_survival = F.logsigmoid(-semantic_logits) + F.logsigmoid(-operational_logits)
            failure = (-torch.expm1(log_survival.float())).clamp(1e-6, 1 - 1e-6)
            risk_logits = torch.logit(failure).to(semantic_logits.dtype)
        # Detach critic predictions: task loss cannot improve routing by merely
        # teaching the critic to claim low risk. Critic learns from outcome labels.
        reliability = F.logsigmoid(-risk_logits.detach())
        routes = ((compatibility + reliability[:, None, :]) / temperature).softmax(-1)
        message = routes @ candidate
        gate = torch.sigmoid(self.gate(torch.cat((candidate, message, anchor), -1)))
        next_state = self.norm(candidate + gate * (message - candidate))
        if return_components:
            return next_state, risk_logits, routes, semantic_logits
        return next_state, risk_logits, routes


class LatentRecursiveReconstruction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.project = nn.Sequential(nn.Linear(c.backbone_dim, c.width), nn.LayerNorm(c.width))
        self.transition = EvidenceGraphTransition(c)
        self.fuse = nn.Linear(2*c.width, c.width)
        self.embedding = nn.Embedding(c.vocab_size, c.width)
        self.decoder = nn.GRU(2*c.width, c.width, c.decoder_layers, batch_first=True)
        self.readout = nn.Linear(c.width, c.vocab_size)

    def encode(self, hidden, mask, dynamic_halt=False):
        if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
            raise ValueError('Expected hidden [batch, prompt_length, backbone_dim] and matching mask')
        mask = mask.bool()
        if not mask.any(-1).all():
            raise ValueError('Each prompt must contain at least one valid token')
        evidence = self.project(hidden.detach())
        pool = (evidence * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        state = pool[:, None, :] + self.transition.roles[None]
        risks, movement = [], []
        active = torch.ones(hidden.shape[0], device=hidden.device, dtype=torch.bool)
        steps = torch.zeros_like(active, dtype=torch.long)
        for index in range(self.config.rounds):
            if self.training and self.config.checkpoint_rounds:
                proposed, risk, routes = checkpoint(self.transition, state, evidence, mask, use_reentrant=False)
            else:
                proposed, risk, routes = self.transition(state, evidence, mask)
            change = (proposed - state).square().mean((1, 2)).sqrt()
            state = torch.where(active[:, None, None], proposed, state)
            steps += active.long()
            risks.append(risk)
            movement.append(change)
            # A stability heuristic, never a correctness certificate. Train with
            # fixed depth so halting cannot cut off learning signals.
            if dynamic_halt and not self.training and index + 1 >= self.config.min_rounds:
                active = active & (change > self.config.halt_tolerance)
                if not active.any():
                    break
        mean = state.mean(1)
        dispersion = state.var(1, unbiased=False)
        context = torch.tanh(self.fuse(torch.cat((mean, dispersion), -1)))
        return context, {'states': state, 'risk_logits': torch.stack(risks, 1),
                         'routes': routes, 'steps': steps, 'movement': torch.stack(movement, 1)}

    def decode(self, context, input_ids, recurrent=None):
        tokens = self.embedding(input_ids)
        conditioning = context[:, None, :].expand(-1, tokens.shape[1], -1)
        if recurrent is None:
            recurrent = context[None].expand(self.config.decoder_layers, -1, -1).contiguous()
        output, recurrent = self.decoder(torch.cat((tokens, conditioning), -1), recurrent)
        return self.readout(output), recurrent

    def forward(self, hidden, mask, decoder_input_ids):
        context, diagnostics = self.encode(hidden, mask)
        logits, _ = self.decode(context, decoder_input_ids)
        return logits, diagnostics

    @torch.no_grad()
    def generate(self, hidden, mask, bos_token_id, eos_token_id, max_new_tokens=128,
                 dynamic_halt=False):
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        was_training = self.training
        self.eval()
        try:
            context, diagnostics = self.encode(hidden, mask, dynamic_halt=dynamic_halt)
            current = torch.full((hidden.shape[0], 1), bos_token_id, dtype=torch.long, device=hidden.device)
            finished = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
            outputs, recurrent = [], None
            for _ in range(max_new_tokens):
                logits, recurrent = self.decode(context, current, recurrent)
                # PAD and BOS cannot be generated as content.
                logits[:, -1, 0] = float('-inf')
                logits[:, -1, bos_token_id] = float('-inf')
                current = logits[:, -1].argmax(-1)
                current = torch.where(finished, eos_token_id, current)
                outputs.append(current)
                finished |= current.eq(eos_token_id)
                current = current[:, None]
                if finished.all():
                    break
            result = torch.stack(outputs, 1) if outputs else current[:, :0]
            return result, diagnostics
        finally:
            self.train(was_training)

    def save(self, path):
        torch.save({'config': asdict(self.config), 'state_dict': self.state_dict()}, path)

    @classmethod
    def load(cls, path, device='cpu'):
        data = torch.load(path, map_location=device, weights_only=True)
        model = cls(LRRConfig(**data['config'])).to(device)
        model.load_state_dict(data.get('model_state', data.get('state_dict')))
        return model


def training_loss(logits, labels, diagnostics, failure_labels=None, critic_weight=0.1):
    """Answer CE plus optional outcome supervision; no invented RL gradient.

    failure_labels: [batch], 1 for failed candidate answers, 0 for passed ones.
    Broadcast supervision is a coarse shared outcome, not node-level truth.
    Only use it when hidden prompts/teacher-forced answers match those outcomes.
    """
    answer = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    critic = answer.new_zeros(())
    if failure_labels is not None:
        risk = diagnostics['risk_logits']
        targets = failure_labels.to(risk)[:, None, None].expand_as(risk)
        critic = F.binary_cross_entropy_with_logits(risk, targets)
    return answer + critic_weight * critic, {'answer_loss': float(answer.detach()), 'critic_loss': float(critic.detach())}


class FrozenBackbone:
    """Optional frozen Hugging Face feature extractor; retains only final states."""
    def __init__(self, model_id, quantize_4bit=False, device='cuda', revision=None,
                 local_files_only=False):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                                       local_files_only=local_files_only)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError('Backbone tokenizer needs a PAD or EOS token')
        options = {'dtype': torch.float16 if device.startswith('cuda') else torch.float32,
                   'revision': revision, 'local_files_only': local_files_only}
        if quantize_4bit:
            from transformers import BitsAndBytesConfig
            if not device.startswith('cuda'):
                raise ValueError('4-bit loading requires CUDA in this implementation')
            options.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                           bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16),
                           device_map={'': device})
        self.model = AutoModel.from_pretrained(model_id, **options)
        if not quantize_4bit:
            self.model.to(device)
        self.model.requires_grad_(False).eval()
        self.device = device
        self.width = self.model.config.hidden_size
        self.revision = getattr(self.model.config, '_commit_hash', None) or revision

    @torch.no_grad()
    def extract(self, prompts, max_length=256, truncate=False):
        inputs = self.tokenizer(prompts, padding=True, truncation=truncate,
                                **({'max_length': max_length} if truncate else {}),
                                return_tensors='pt').to(self.device)
        if inputs.input_ids.shape[1] > max_length:
            raise ValueError('Prompt exceeds --max-prompt-tokens; raise limit or use --truncate-prompts')
        output = self.model(**inputs, use_cache=False, output_hidden_states=False)
        return output.last_hidden_state.detach(), inputs.attention_mask.bool()


def smoke_test():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = LatentRecursiveReconstruction(LRRConfig(backbone_dim=16, vocab_size=12, width=24, nodes=3, rounds=3))
    hidden = torch.randn(4, 5, 16, requires_grad=True)
    mask = torch.tensor([[1,1,1,0,0]]*4, dtype=torch.bool)
    inputs = torch.tensor([[1,3,4]]*4)
    labels = torch.tensor([[3,4,2]]*4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    losses = []
    for _ in range(35):
        optimizer.zero_grad()
        logits, info = model(hidden, mask, inputs)
        loss, _ = training_loss(logits, labels, info, torch.zeros(4))
        loss.backward()
        assert hidden.grad is None, 'Backbone representations must be detached'
        assert model.transition.propose[0].weight.grad.abs().sum() > 0
        assert model.transition.edge[0].weight.grad.abs().sum() > 0
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, losses
    model.eval()
    with torch.no_grad():
        original, info = model(hidden, mask, inputs)
        changed = hidden.detach().clone()
        changed[:, 3:] = 1000
        masked, _ = model(changed, mask, inputs)
        torch.testing.assert_close(original, masked)
        torch.testing.assert_close(info['routes'].sum(-1), torch.ones(4,3))
    model.train()
    model.config.checkpoint_rounds = True
    checkpoint_logits, checkpoint_info = model(hidden, mask, inputs)
    training_loss(checkpoint_logits, labels, checkpoint_info)[0].backward()
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model.pt'
        model.save(path)
        restored = LatentRecursiveReconstruction.load(path).eval()
        model.eval()
        torch.testing.assert_close(model(hidden, mask, inputs)[0], restored(hidden, mask, inputs)[0])
    tokens, info = model.generate(hidden, mask, 1, 2, 8)
    assert tokens.shape[0] == 4 and info['steps'].max() <= 3
    print(json.dumps({'smoke_test': 'passed', 'initial_loss': losses[0], 'final_loss': losses[-1],
                      'trainable_parameters': sum(p.numel() for p in model.parameters())}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        parser.print_help()
