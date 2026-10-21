# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Action relevance for GR00T N1.7: DiT action-token -> image-token cross-attention.

GR00T has no action tokens in the LLM; actions come from the AlternateVLDiT, whose blocks
cross-attend to the backbone features (image tokens and text in alternating blocks). The
analogue of VLA-Pruner's action-decode attention (paper Eq. 3-4, pi0: averaged over
flow-matching steps) is the attention of the DiT action tokens on the image tokens:

    A_act = softmax(Q_action K_vl^T / sqrt(d))   per image cross-attention block and step
    S_act[m] = mean over heads, action tokens, selected blocks and steps of A_act[:, m]

Forward pre-hooks on the selected blocks' ``attn1`` recompute the probabilities exactly as
diffusers' AttnProcessor2_0 does (the SDPA kernel returns no weights).
"""

from gr00t.model.modules.dit import AlternateVLDiT
from gr00t_efficientvla.dit_cache import denoising_step
import torch


def image_cross_attention_blocks(dit: AlternateVLDiT) -> list[int]:
    """Indices of the blocks that cross-attend to image tokens (see AlternateVLDiT.forward)."""
    period = 2 * dit.attend_text_every_n_blocks
    return [i for i in range(len(dit.transformer_blocks)) if i % 2 == 0 and i % period != 0]


def cross_attention_probs(attn, hidden_states, encoder_hidden_states, attention_mask):
    """(B, heads, queries, keys) float32 probabilities of a diffusers ``Attention`` call."""
    query = attn.to_q(hidden_states)
    if attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
    key = attn.to_k(encoder_hidden_states)
    batch = hidden_states.shape[0]
    head_dim = key.shape[-1] // attn.heads
    query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * head_dim**-0.5
    if attention_mask is not None:
        logits = logits.masked_fill(~attention_mask[:, None, None, :].bool(), float("-inf"))
    return logits.softmax(dim=-1)


class ActionAttentionRecorder:
    """Accumulates S_act over one ``get_action`` call; ``pop`` returns and clears it."""

    def __init__(self, action_head, blocks: str = "latter_half", steps: str = "all"):
        dit = action_head.model
        if not isinstance(dit, AlternateVLDiT):
            raise TypeError(f"expected AlternateVLDiT, got {type(dit).__name__}")
        image_blocks = image_cross_attention_blocks(dit)
        if blocks == "latter_half":
            image_blocks = [i for i in image_blocks if i >= len(dit.transformer_blocks) // 2]
        self.blocks = image_blocks
        self.num_steps = action_head.num_inference_timesteps
        self.num_buckets = action_head.num_timestep_buckets
        self.action_horizon = action_head.action_horizon
        self.steps = steps
        self.step = 0
        self.total = None
        self.count = 0
        self.handles = [dit.register_forward_pre_hook(self._on_dit, with_kwargs=True)]
        for i in self.blocks:
            attn = dit.transformer_blocks[i].attn1
            self.handles.append(
                attn.register_forward_pre_hook(self._on_attention, with_kwargs=True)
            )

    def _on_dit(self, _module, _args, kwargs):
        self.step = denoising_step(kwargs["timestep"], self.num_steps, self.num_buckets)

    def _use_step(self) -> bool:
        return self.steps == "all" or self.step >= self.num_steps // 2

    @torch.no_grad()
    def _on_attention(self, attn, args, kwargs):
        if not self._use_step():
            return
        hidden_states = args[0] if args else kwargs["hidden_states"]
        probs = cross_attention_probs(
            attn,
            hidden_states[:, -self.action_horizon :],  # action tokens; the state token is skipped
            kwargs["encoder_hidden_states"],
            kwargs.get("attention_mask"),
        )
        scores = probs.mean(dim=(1, 2))  # (B, keys)
        self.total = scores if self.total is None else self.total + scores
        self.count += 1

    def pop(self) -> torch.Tensor | None:
        """Mean action -> key attention (B, keys) of the last ``get_action``, or None."""
        if self.count == 0:
            return None
        out = self.total / self.count
        self.total, self.count = None, 0
        return out

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
