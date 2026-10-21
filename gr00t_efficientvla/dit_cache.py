# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EfficientVLA static N-step caching of DiT attention / MLP outputs (Sec. 3.4).

On denoising steps with ``step % interval == 0`` every DiT block computes its attention and
feed-forward outputs and stores them; on the other steps both are skipped and the stored
outputs are added to the block's current residual stream. Everything outside the blocks
(action encoder, timestep embedding, output projection, decoder, Euler update) runs every step.

The denoising step is recovered from the DiT ``timestep`` argument, which the action head sets
to ``int(step / num_steps * num_buckets)``, so no change to the denoising loop is needed. Step
0 always recomputes, so the cache never leaks across ``get_action`` calls.
"""

from gr00t.model.modules.dit import BasicTransformerBlock, DiT, _sdpa_context
import torch


class CachedBasicTransformerBlock(BasicTransformerBlock):
    _evla_cache_state: dict
    _evla_cached: tuple[torch.Tensor, torch.Tensor] | None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        temb=None,
    ):
        if not self._evla_cache_state["compute"]:
            attn_output, ff_output = self._evla_cached
            hidden_states = attn_output + hidden_states
            return ff_output + hidden_states

        # Same as BasicTransformerBlock.forward, keeping the two residual branches.
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)
        with _sdpa_context():
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=(
                    encoder_attention_mask if encoder_hidden_states is not None else attention_mask
                ),
            )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        ff_output = self.ff(self.norm3(hidden_states))
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        self._evla_cached = (attn_output, ff_output)
        return hidden_states


def denoising_step(timestep: torch.Tensor, num_steps: int, num_buckets: int) -> int:
    return round(float(timestep.reshape(-1)[0]) * num_steps / num_buckets)


def enable_dit_cache(action_head, interval: int) -> None:
    """Cache DiT block outputs every ``interval`` denoising steps (1 keeps every step exact)."""
    disable_dit_cache(action_head)
    dit = action_head.model
    if not isinstance(dit, DiT):
        raise TypeError(f"expected a DiT action model, got {type(dit).__name__}")
    state = {"compute": True, "interval": interval, "step": None}
    num_steps = action_head.num_inference_timesteps
    num_buckets = action_head.num_timestep_buckets

    def pre_hook(_module, _args, kwargs):
        step = denoising_step(kwargs["timestep"], num_steps, num_buckets)
        state["step"] = step
        state["compute"] = step % state["interval"] == 0

    for block in dit.transformer_blocks:
        block.__class__ = CachedBasicTransformerBlock
        block._evla_cache_state = state
        block._evla_cached = None
    dit._evla_cache_hook = dit.register_forward_pre_hook(pre_hook, with_kwargs=True)
    dit._evla_cache_state = state


def disable_dit_cache(action_head) -> None:
    dit = action_head.model
    hook = getattr(dit, "_evla_cache_hook", None)
    if hook is not None:
        hook.remove()
        dit._evla_cache_hook = None
    for block in dit.transformer_blocks:
        if isinstance(block, CachedBasicTransformerBlock):
            block.__class__ = BasicTransformerBlock
            block._evla_cached = None
