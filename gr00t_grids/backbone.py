# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3 backbone with the Differentiable Grid Sampler between the vision merger and the LLM.

Per image, the merged token map (h' x w' x D, 8 x 8 for 256 px) goes through the sampler, which
predicts K coordinates from the pooled map and bilinearly reads K tokens (plus a coordinate
embedding). The sequence keeps the first K of the image's placeholder positions and the LLM,
and then the action head, see only those K tokens per image. DeepStack maps are read at the
same coordinates. MRoPE ids of the sampled tokens follow ``GridSConfig.position_mode``; text
keeps its original positions.
"""

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
from gr00t_adp.config import ADPConfig
from gr00t_adp.qwen3_backbone_adp import Qwen3BackboneADP
import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from .config import GridSConfig
from .sampler import ActiveTokenSampler, grid_sample_bilinear


class Qwen3BackboneGridS(Qwen3BackboneADP):
    grids_config: GridSConfig
    grid_sampler: nn.ModuleList
    grids_last: dict | None
    # Tests only: fixed (n_images, K, 2) coordinates instead of the predicted ones.
    grids_coords_override: torch.Tensor | None

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        prepared = self.prepare_sampled_inputs(vl_input)
        hidden = self.run_language_model(prepared)
        return BatchFeature(
            data={
                "backbone_features": hidden,
                "backbone_attention_mask": prepared["attention_mask"] == 1,
                "image_mask": prepared["image_mask"],
            }
        )  # [B, T2, hidden_size]

    def prepare_sampled_inputs(self, vl_input: BatchFeature) -> dict:
        cfg = self.grids_config
        input_ids = vl_input["input_ids"]
        attention_mask = vl_input["attention_mask"]
        image_grid_thw = vl_input["image_grid_thw"]
        core = self.model.model  # Qwen3VLModel
        merge = core.visual.spatial_merge_size
        batch = input_ids.shape[0]
        k = cfg.num_tokens

        inputs_embeds = core.get_input_embeddings()(input_ids)
        image_embeds, deepstack = core.get_image_features(vl_input["pixel_values"], image_grid_thw)
        image_mask_full = input_ids == self.model.config.image_token_id
        position_ids, _ = core.get_rope_index(
            input_ids, image_grid_thw, None, attention_mask=attention_mask
        )

        n_images = len(image_embeds)
        assert n_images % batch == 0, "samples must have the same number of images"
        per_sample = n_images // batch
        grids = image_grid_thw.tolist()
        assert all(t == 1 for t, _, _ in grids), "one temporal patch per image is assumed"
        sizes = [(h // merge, w // merge) for _, h, w in grids]
        if any(h * w < k for h, w in sizes):
            raise ValueError(f"num_tokens={k} exceeds the tokens of an image {sizes}")

        # Sample every image: K tokens, their coordinates, and K rows of each DeepStack map.
        ds_offsets = torch.tensor([0] + [h * w for h, w in sizes], device=input_ids.device).cumsum(
            0
        )
        sampled, coords_all, ds_sampled = [], [], [[] for _ in deepstack]
        for i, (feat, (h, w)) in enumerate(zip(image_embeds, sizes)):
            # Images are frame-major within a sample: (t0, v0), (t0, v1), ..., (t1, v0), ...
            view = (i % per_sample) % len(self.grid_sampler)
            sampler = self.grid_sampler[0 if cfg.share_across_views else view]
            override = None
            if self.grids_coords_override is not None:
                override = self.grids_coords_override[i : i + 1].to(feat.device)
            tokens, coords = sampler(feat.view(1, h, w, -1), override)
            sampled.append(tokens[0].to(inputs_embeds.dtype))
            coords_all.append(coords[0])
            if cfg.sample_deepstack:
                for level, d in enumerate(deepstack):
                    rows = d[ds_offsets[i] : ds_offsets[i + 1]].view(1, h, w, -1)
                    ds_sampled[level].append(
                        grid_sample_bilinear(rows.to(coords.dtype), coords)[0].to(d.dtype)
                    )

        # Keep the first K placeholder positions of every image.
        keep = ~image_mask_full
        new_position_ids = position_ids.clone().float()
        for b in range(batch):
            image_pos = image_mask_full[b].nonzero().squeeze(-1)
            offset = 0
            for j in range(per_sample):
                i = b * per_sample + j
                h, w = sizes[i]
                pos = image_pos[offset : offset + h * w]
                kept = pos[:k]
                keep[b, kept] = True
                # Coordinates are detached here: as in GridS, they are learned only through the
                # bilinear weights and the coordinate embedding, not through RoPE.
                new_position_ids[:, b, kept] = self._sampled_positions(
                    position_ids[:, b, pos], coords_all[i].detach(), h, w
                )
                offset += h * w

        idx = keep.nonzero()[:, 1].view(batch, -1)  # (B, L'), ascending
        inputs_embeds = torch.gather(
            inputs_embeds, 1, idx[..., None].expand(-1, -1, inputs_embeds.shape[-1])
        )
        image_mask = torch.gather(image_mask_full, 1, idx)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask[..., None].expand_as(inputs_embeds), torch.cat(sampled)
        )
        prepared = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": torch.gather(attention_mask, 1, idx),
            "position_ids": torch.gather(new_position_ids, 2, idx[None].expand(3, -1, -1)),
            "image_mask": image_mask,
            "deepstack_embeds": [torch.cat(d) for d in ds_sampled]
            if cfg.sample_deepstack
            else None,
        }
        self.grids_last = {"coords": torch.stack(coords_all).detach(), "sizes": sizes}
        return prepared

    def _sampled_positions(self, positions, coords, h, w) -> torch.Tensor:
        """MRoPE ids (3, K) of the K sampled tokens of one image with full-grid ids ``positions``."""
        cfg = self.grids_config
        k = cfg.num_tokens
        t0, h0, w0 = positions[0].min(), positions[1].min(), positions[2].min()
        out = torch.empty(3, k, dtype=torch.float32, device=positions.device)
        out[0] = t0
        if cfg.position_mode == "continuous":
            out[1] = h0 + coords[:, 1].float() * (h - 1)
            out[2] = w0 + coords[:, 0].float() * (w - 1)
        elif cfg.position_mode == "grid":
            side = round(k**0.5)
            r = torch.arange(k, device=positions.device)
            out[1] = h0 + torch.div(r, side, rounding_mode="floor")
            out[2] = w0 + r % side
        else:  # "first"
            out = positions[:, :k].float()
        return out


def build_samplers(config: GridSConfig, vision_dim: int) -> nn.ModuleList:
    n = 1 if config.share_across_views else config.num_views
    return nn.ModuleList(
        ActiveTokenSampler(
            vision_dim,
            num_tokens=config.num_tokens,
            embed_coords=config.embed_coords,
            input_layernorm=config.input_layernorm,
            scout_hidden=config.scout_hidden,
            coord_hidden=config.coord_hidden,
        )
        for _ in range(n)
    )


def enable_grids(
    backbone: Qwen3Backbone, config: GridSConfig, samplers: nn.ModuleList | None = None
):
    """Attach a (new, trainable) Grid Sampler to a loaded ``Qwen3Backbone`` in place."""
    if not isinstance(backbone, Qwen3Backbone):
        raise TypeError(f"expected Qwen3Backbone, got {type(backbone).__name__}")
    if samplers is None:
        vision_dim = backbone.model.config.text_config.hidden_size
        param = next(backbone.parameters())
        samplers = build_samplers(config, vision_dim).to(param.device)
    backbone.__class__ = Qwen3BackboneGridS
    backbone.grids_config = config
    backbone.grid_sampler = samplers  # registered as a submodule: saved with the model
    backbone.grids_last = None
    backbone.grids_coords_override = None
    # Bookkeeping helpers inherited from the ADP backbone read this; its pruning stays off.
    backbone.adp_config = ADPConfig(keep_ratio=1.0)
    backbone.adp_last = None
    return backbone
