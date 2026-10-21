# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrate LLM layer importance I(l) = 1 - mean cos(x_l, x_{l+1}) (EfficientVLA Eq. 1).

The cosine is between each decoder layer's input and output hidden state (before any DeepStack
addition), averaged over the non-padding tokens of a sample and then over samples. Samples are
evenly spaced steps of the first ``--num-episodes`` episodes. Writes a JSON with the per-layer
importance and the ranking (least important first) used by ``drop_layers_from_importance``.

    python -m gr00t_efficientvla.layer_importance --model-path <ckpt> --dataset-path <dataset> \
        --output-json <out_dir>/layer_importance.json
"""

from dataclasses import dataclass
import json

from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t_adp.adp_policy import no_flash_attention
from gr00t_adp.data_utils import build_model_inputs, load_observations
import numpy as np
import torch
import torch.nn.functional as F
import tyro


@dataclass
class Args:
    model_path: str
    dataset_path: str
    output_json: str
    embodiment_tag: str = "new_embodiment"
    num_episodes: int = 20
    steps_per_episode: int = 5
    device: str = "cuda"


@torch.inference_mode()
def layer_importance(policy: Gr00tPolicy, observations: list[dict], dtype) -> np.ndarray:
    backbone = policy.model.backbone
    layers = backbone.model.model.language_model.layers
    per_sample = []
    current = {}

    def make_hook(idx):
        def hook(_module, args, output):
            x_in, x_out = args[0].float(), output.float()
            cos = F.cosine_similarity(x_in, x_out, dim=-1)  # (B, L)
            valid = current["valid"]
            current["cos"][:, idx] = (cos * valid).sum(-1) / valid.sum(-1)

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        for obs in observations:
            inputs, _ = build_model_inputs(policy, obs, dtype=dtype)
            mask = inputs["attention_mask"]
            current["valid"] = (mask == 1).float()
            current["cos"] = torch.zeros(mask.shape[0], len(layers), device=mask.device)
            backbone(inputs)
            per_sample.append(current["cos"].cpu())
    finally:
        for h in handles:
            h.remove()
    return 1.0 - torch.cat(per_sample).mean(dim=0).numpy()


def main(args: Args):
    device = torch.device(args.device)
    if device.type == "cpu":
        with no_flash_attention():
            policy = Gr00tPolicy(args.embodiment_tag, args.model_path, device=args.device)
    else:
        policy = Gr00tPolicy(args.embodiment_tag, args.model_path, device=args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    policy.model.to(dtype)

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    loader = LeRobotEpisodeLoader(args.dataset_path, modality_configs=policy.get_modality_config())
    observations, samples = [], []
    for ep in range(min(args.num_episodes, len(loader))):
        length = loader.get_episode_length(ep)
        steps = np.linspace(0, length - 1, args.steps_per_episode).astype(int).tolist()
        observations += load_observations(policy, args.dataset_path, ep, steps)
        samples += [[ep, s] for s in steps]
    print(f"{len(observations)} calibration samples from {args.num_episodes} episodes")

    importance = layer_importance(policy, observations, dtype)
    ranking = [int(i) for i in np.argsort(importance, kind="stable")]
    result = {
        "importance": [float(v) for v in importance],
        "ranking": ranking,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "samples": samples,
    }
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    for i, v in enumerate(importance):
        print(f"layer {i:2d}: I={v:.4f}")
    print(f"ranking (least important first): {ranking}\nsaved {args.output_json}")


if __name__ == "__main__":
    main(tyro.cli(Args))
