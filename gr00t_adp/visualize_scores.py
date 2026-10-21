# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Overlay ADP text-to-image scores and the kept tokens on the frames the VLM sees.

One figure per trajectory step: row 1 is the score heatmap per view, row 2 darkens the
tokens that would be dropped at ``--keep-ratio``.

    python -m gr00t_adp.visualize_scores --model-path <ckpt> --dataset-path <dataset> \
        --steps 0 80 160 --keep-ratio 0.5 --out-dir <dir>
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402

from gr00t_adp import ADPConfig  # noqa: E402
from gr00t_adp.adp_policy import Gr00tADPPolicy, no_flash_attention  # noqa: E402
from gr00t_adp.data_utils import build_model_inputs, load_observations  # noqa: E402


@dataclass
class Args:
    model_path: str
    dataset_path: str
    out_dir: str
    embodiment_tag: str = "new_embodiment"
    traj_id: int = 0
    steps: tuple[int, ...] = (0,)
    keep_ratio: float = 0.5
    budget_policy: str = "per_view_uniform"
    text_source: str = "task"
    device: str = "cpu"


def token_grid(values: torch.Tensor, grid_thw: list[int], merge: int) -> np.ndarray:
    t, h, w = grid_thw
    return values.float().view(t, h // merge, w // merge)[0].numpy()


def plot_step(images, record, merge, keep_ratio, title, path):
    n_views = len(images)
    fig, axes = plt.subplots(2, n_views, figsize=(3.2 * n_views, 6.6), squeeze=False)
    scores, kept = record["scores"][0], record["kept"][0]
    vmin = min(float(s.min()) for s in scores)
    vmax = max(float(s.max()) for s in scores)
    for v in range(n_views):
        img = images[v].permute(1, 2, 0).numpy()
        height, width = img.shape[:2]
        heat = token_grid(scores[v], record["grid_thw"][v], merge)
        keep = torch.zeros(scores[v].numel(), dtype=torch.bool)
        keep[kept[v]] = True
        keep_grid = token_grid(keep, record["grid_thw"][v], merge)
        extent = (0, width, height, 0)
        name = record["views"][v] if record["views"] else f"image {v}"

        ax = axes[0, v]
        ax.imshow(img)
        im = ax.imshow(
            heat,
            cmap="inferno",
            alpha=0.55,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(name, fontsize=9)
        ax.axis("off")

        ax = axes[1, v]
        shade = np.where(keep_grid[..., None], 0.0, 0.75) * np.ones(3)
        ax.imshow(img)
        ax.imshow(
            np.concatenate([np.zeros_like(shade), shade[..., :1]], axis=-1),
            extent=extent,
            interpolation="nearest",
        )
        ax.set_title(f"kept {int(keep.sum())}/{keep.numel()}", fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.01, label="mean Q·K logit")
    fig.suptitle(f"{title}  (keep_ratio={keep_ratio})", fontsize=10)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(args: Args):
    config = ADPConfig(
        keep_ratio=args.keep_ratio,
        budget_policy=args.budget_policy,
        text_source=args.text_source,
        record=True,
    )
    with no_flash_attention():
        policy = Gr00tADPPolicy(
            args.embodiment_tag, args.model_path, device=args.device, adp_config=config
        )
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    policy.model.to(dtype)
    backbone = policy.backbone
    merge = backbone.model.model.visual.spatial_merge_size
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    observations = load_observations(policy, args.dataset_path, args.traj_id, list(args.steps))
    for step, obs in zip(args.steps, observations):
        (inputs, _), images = build_model_inputs(policy, obs, dtype=dtype, return_images=True)
        with torch.inference_mode():
            backbone(inputs)
        text = next(iter(obs["language"].values()))[0][0]
        path = out_dir / f"traj{args.traj_id}_step{step:04d}.png"
        plot_step(
            images[0],
            backbone.adp_last,
            merge,
            args.keep_ratio,
            f"traj {args.traj_id} step {step}: {text}",
            path,
        )
        print(f"saved {path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
