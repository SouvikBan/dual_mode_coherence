"""Batched locally typical sampling for vLLM V1."""

import torch
from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates


def typical_filter(logits, masses):
    """Filter rows by typical probability mass; masses has shape (rows, 1)."""
    scores = logits.float()
    log_probs = scores.log_softmax(dim=-1)
    entropy = -(log_probs * log_probs.exp()).nansum(dim=-1, keepdim=True)

    # Rank tokens by distance between their surprisal and the entropy.
    deviation = (-log_probs - entropy).abs()
    ranked_deviation, order = deviation.sort(dim=-1)
    cumulative = scores.gather(-1, order).softmax(dim=-1).cumsum(dim=-1)

    # Include the token crossing the mass threshold and all boundary ties.
    # This also guarantees at least one retained token, as in Transformers.
    boundary = (cumulative < masses).sum(dim=-1, keepdim=True)
    boundary = boundary.clamp(max=scores.shape[-1] - 1)
    cutoff = ranked_deviation.gather(-1, boundary)
    remove = (deviation > cutoff) & (masses < 1.0)
    return logits.masked_fill(remove, float("-inf"))


def typical_mass(params, prompt_ids=None, output_ids=None):
    """Return this request's mass, or None for an ordinary sampling request."""
    mass = float((params.extra_args or {}).get("typical_p", 1.0))
    if not 0.0 < mass <= 1.0:
        raise ValueError("typical_p must be in (0, 1]")
    if mass < 1.0 and params.temperature != 1.0:
        # vLLM applies this processor before its temperature scaling.
        raise ValueError("These typical strategies require temperature=1.0")
    return mass if mass < 1.0 else None


class TypicalLogitsProcessor(LogitsProcessor):
    """Track request masses as vLLM adds, removes, and reorders batch rows."""

    def __init__(self, vllm_config, device, is_pin_memory):
        self.device = device
        self.req_info = {}
        self.rows = None
        self.masses = None

    @classmethod
    def validate_params(cls, sampling_params):
        typical_mass(sampling_params)

    def is_argmax_invariant(self):
        # Typical sampling can exclude the most probable token.
        return False

    def update_state(self, batch_update):
        if batch_update is None:
            return
        process_dict_updates(self.req_info, batch_update, typical_mass)
        if self.req_info:
            self.rows = torch.tensor(
                list(self.req_info), dtype=torch.long, device=self.device
            )
            self.masses = torch.tensor(
                list(self.req_info.values()), dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1)
        else:
            self.rows = self.masses = None

    def apply(self, logits):
        if self.rows is None:
            return logits
        # Bound temporary sorting memory for Gemma's large vocabulary.
        # Each chunk is processed together on the GPU, without per-row calls.
        for start in range(0, self.rows.numel(), 32):
            rows = self.rows[start:start + 32]
            scores = logits.index_select(0, rows)
            filtered = typical_filter(scores, self.masses[start:start + 32])
            logits.index_copy_(0, rows, filtered)
        return logits
