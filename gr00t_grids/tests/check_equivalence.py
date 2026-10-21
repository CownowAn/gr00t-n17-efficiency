# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checks of the Grid Sampler integration on a real GR00T N1.7 checkpoint (CPU, fp32).

1. Identity: K = all tokens, coordinates fixed on the token grid, no coordinate embedding ->
   features and action_pred equal the original model (checks the sequence gather, MRoPE ids,
   DeepStack sampling and masks) for every position mode.
2. K < N with a fresh sampler: K tokens per image reach the LLM, MRoPE ids follow the mode,
   padded batches work, outputs are finite.
3. Training signal: gradients reach every sampler parameter through the frozen LLM while the
   frozen LLM / ViT receive none; with per-view samplers each one is trained.
4. Save / load: a checkpoint carrying ``backbone.grid_sampler.*`` weights and
   ``experiment_cfg/grids_config.json`` loads through ``Gr00tGridSPolicy`` with identical outputs.

    CUDA_VISIBLE_DEVICES="" python -m gr00t_grids.tests.check_equivalence \
        --model-path <ckpt> --dataset-path <lerobot dataset> --work-dir <scratch dir>
"""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t_adp.adp_policy import no_flash_attention
from gr00t_adp.data_utils import build_model_inputs, load_observations, stack_observations
from safetensors.torch import save_file
import torch
from transformers.feature_extraction_utils import BatchFeature
import tyro

from gr00t_grids import Gr00tGridSPolicy, GridSConfig
from gr00t_grids.backbone import enable_grids
from gr00t_grids.policy import SAMPLER_PREFIX


@dataclass
class Args:
    model_path: str
    dataset_path: str
    work_dir: str
    embodiment_tag: str = "new_embodiment"
    traj_id: int = 0
    steps: tuple[int, ...] = (0, 40)
    num_tokens: int = 16


TOL = 1e-4


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def fresh(backbone_output):
    """Shallow copy: the action head overwrites ``backbone_features`` in place."""
    return BatchFeature(data=dict(backbone_output))


def restore(backbone):
    backbone.__class__ = Qwen3Backbone
    if "grid_sampler" in backbone._modules:
        del backbone._modules["grid_sampler"]


def grid_coords(n_images: int, h: int, w: int) -> torch.Tensor:
    r, c = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    coords = torch.stack([c.flatten().double() / (w - 1), r.flatten().double() / (h - 1)], -1)
    return coords[None].expand(n_images, -1, -1)


class Checker:
    def __init__(self, policy: Gr00tPolicy):
        self.policy = policy
        self.model = policy.model
        self.backbone = policy.model.backbone
        self.n_views = len(policy.modality_configs["video"].modality_keys)

    def action(self, out, action_inputs):
        torch.manual_seed(0)
        return self.model.action_head.get_action(fresh(out), action_inputs)["action_pred"]

    def original(self, inputs):
        restore(self.backbone)
        return Qwen3Backbone.forward(self.backbone, inputs)

    # 1 -----------------------------------------------------------------------------------
    def check_identity(self, inputs, action_inputs) -> bool:
        ref = self.original(inputs)
        ref_act = self.action(ref, action_inputs)
        n_images = inputs["image_grid_thw"].shape[0]
        _, h, w = (inputs["image_grid_thw"][0] // torch.tensor([1, 2, 2])).tolist()
        ok = True
        for mode in ("continuous", "first"):
            cfg = GridSConfig(num_tokens=h * w, embed_coords=False, position_mode=mode)
            enable_grids(self.backbone, cfg)
            self.backbone.grids_coords_override = grid_coords(n_images, h, w)
            out = self.backbone(inputs)
            valid = ref["backbone_attention_mask"]
            same = torch.equal(valid, out["backbone_attention_mask"])
            same &= torch.equal(ref["image_mask"], out["image_mask"])
            err = max_rel(out["backbone_features"][valid], ref["backbone_features"][valid])
            err_a = max_rel(self.action(out, action_inputs), ref_act)
            ok &= report(
                f"identity sampling == original ({mode})",
                same and err < TOL and err_a < TOL,
                f"features rel err={err:.3e}, action_pred rel err={err_a:.3e}",
            )
        self.backbone.grids_coords_override = None
        return ok

    # 2 -----------------------------------------------------------------------------------
    def check_sampled(self, inputs, action_inputs, k) -> bool:
        ref = self.original(inputs)
        ok = True
        for mode in ("continuous", "grid", "first"):
            if mode == "grid" and round(k**0.5) ** 2 != k:
                continue
            torch.manual_seed(0)
            enable_grids(self.backbone, GridSConfig(num_tokens=k, position_mode=mode))
            prepared = self.backbone.prepare_sampled_inputs(inputs)
            out = self.backbone(inputs)
            n_images = inputs["image_grid_thw"].shape[0] // inputs["input_ids"].shape[0]
            n_img_tokens = out["image_mask"].sum(dim=1)
            good = bool((n_img_tokens == n_images * k).all())
            removed = ref["backbone_features"].shape[1] - out["backbone_features"].shape[1]
            good &= removed == n_images * (64 - k)
            # Sampled positions of the first image of sample 0 stay inside its 8 x 8 block.
            pos = prepared["position_ids"][:, 0][:, prepared["image_mask"][0]][:, :k]
            full_pos = self.backbone.model.model.get_rope_index(
                inputs["input_ids"], inputs["image_grid_thw"], None, inputs["attention_mask"]
            )[0][:, 0][:, ref["image_mask"][0]][:, :64].float()
            inside = bool(
                (
                    (pos[1:] >= full_pos[1:].min(1, keepdim=True).values - 1e-4)
                    & (pos[1:] <= full_pos[1:].max(1, keepdim=True).values + 1e-4)
                ).all()
            )
            fractional = bool((pos[1:] != pos[1:].round()).any())
            act = self.action(out, action_inputs)
            good &= inside and bool(torch.isfinite(act).all())
            ok &= report(
                f"K={k} sampling ({mode})",
                good,
                f"image tokens per sample {n_img_tokens.tolist()}, seq "
                f"{ref['backbone_features'].shape[1]}->{out['backbone_features'].shape[1]}, "
                f"positions inside image block={inside}, fractional={fractional}",
            )
        return ok

    # 3 -----------------------------------------------------------------------------------
    def check_gradients(self, inputs, k) -> bool:
        ok = True
        for share in (True, False):
            torch.manual_seed(0)
            cfg = GridSConfig(num_tokens=k, share_across_views=share, num_views=self.n_views)
            enable_grids(self.backbone, cfg)
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out = self.backbone(inputs)
                feats = out["backbone_features"][out["backbone_attention_mask"]]
                feats.float().pow(2).mean().backward()
            sampler_grads = [
                all(p.grad is not None and p.grad.abs().sum() > 0 for p in s.parameters())
                for s in self.backbone.grid_sampler
            ]
            frozen = [
                p
                for n, p in self.backbone.named_parameters()
                if "grid_sampler" not in n and not p.requires_grad
            ]
            leaked = sum(p.grad is not None for p in frozen)
            ok &= report(
                f"gradients ({'shared' if share else 'per-view'} sampler)",
                all(sampler_grads) and leaked == 0,
                f"samplers with gradient {sum(sampler_grads)}/{len(sampler_grads)}, "
                f"frozen backbone tensors {len(frozen)} with gradient {leaked}",
            )
            self.model.zero_grad(set_to_none=True)
        return ok

    # 4 -----------------------------------------------------------------------------------
    def check_save_load(
        self, model_path, dataset_path, embodiment_tag, inputs, k, work_dir
    ) -> bool:
        torch.manual_seed(0)
        cfg = GridSConfig(num_tokens=k)
        enable_grids(self.backbone, cfg)
        with torch.no_grad():  # make the weights distinguishable from a fresh init
            for p in self.backbone.grid_sampler.parameters():
                p.add_(0.01 * torch.randn_like(p))
        expected = self.backbone(inputs)["backbone_features"]

        src, dst = Path(model_path), Path(work_dir) / "grids_ckpt"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for item in src.iterdir():
            if item.name in (
                "model.safetensors.index.json",
                "experiment_cfg",
            ) or item.name.startswith("checkpoint-"):
                continue
            (dst / item.name).symlink_to(item)
        (dst / "experiment_cfg").mkdir()
        for item in (src / "experiment_cfg").iterdir():
            (dst / "experiment_cfg" / item.name).symlink_to(item)
        cfg.save(dst / "experiment_cfg")
        state = {
            SAMPLER_PREFIX + n: t.detach().clone().contiguous()
            for n, t in self.backbone.grid_sampler.state_dict().items()
        }
        save_file(state, str(dst / "model-grids.safetensors"))
        index = json.loads((src / "model.safetensors.index.json").read_text())
        index["weight_map"].update({key: "model-grids.safetensors" for key in state})
        (dst / "model.safetensors.index.json").write_text(json.dumps(index))

        with no_flash_attention():
            loaded = Gr00tGridSPolicy(embodiment_tag, str(dst), device="cpu")
        loaded.model.to(torch.float32)
        got = loaded.backbone(inputs)["backbone_features"]
        err = max_rel(got, expected)
        same_cfg = loaded.grids_config == cfg
        del loaded
        return report(
            "save / load through Gr00tGridSPolicy",
            err < TOL and same_cfg,
            f"features rel err={err:.3e}, config restored={same_cfg}",
        )


def main(args: Args):
    with no_flash_attention():
        policy = Gr00tPolicy(args.embodiment_tag, args.model_path, device="cpu")
    policy.model.to(torch.float32)
    checker = Checker(policy)
    print(f"views={list(policy.modality_configs['video'].modality_keys)}, K={args.num_tokens}")

    observations = load_observations(policy, args.dataset_path, args.traj_id, list(args.steps))
    second = observations[1]
    lang_key = next(iter(second["language"]))
    second["language"][lang_key] = [[second["language"][lang_key][0][0] + " carefully and quickly"]]
    batches = {"B=1": observations[0], "B=2 (padded)": stack_observations(observations[:2])}

    all_ok = True
    # no_grad, not inference_mode: the gradient check reuses these tensors.
    with torch.no_grad():
        built = {
            name: build_model_inputs(policy, obs, dtype=torch.float32)
            for name, obs in batches.items()
        }
    for name, (inputs, action_inputs) in built.items():
        print(f"--- {name}, sequence {tuple(inputs['input_ids'].shape)}")
        with torch.no_grad():
            all_ok &= checker.check_identity(inputs, action_inputs)
            all_ok &= checker.check_sampled(inputs, action_inputs, args.num_tokens)
        all_ok &= checker.check_gradients(inputs, args.num_tokens)
    with torch.no_grad():
        all_ok &= checker.check_save_load(
            args.model_path,
            args.dataset_path,
            args.embodiment_tag,
            built["B=1"][0],
            args.num_tokens,
            args.work_dir,
        )
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main(tyro.cli(Args))
