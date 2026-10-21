# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers to pull observations from a LeRobot dataset and build model inputs (as in open_loop_eval)."""

from copy import deepcopy

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import MessageType
from gr00t.data.utils import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
import numpy as np
import torch


def load_observations(
    policy: Gr00tPolicy, dataset_path: str, traj_id: int, steps: list[int]
) -> list[dict]:
    """One batched (B=1) policy observation per requested step of a trajectory."""
    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=dataset_path, modality_configs=modality)
    traj = loader[traj_id]
    obs_configs = deepcopy(loader.modality_configs)
    obs_configs.pop("action")
    observations = []
    for step in steps:
        data_point = extract_step_data(traj, step, obs_configs, policy.embodiment_tag)
        obs = {f"state.{k}": v for k, v in data_point.states.items()}
        obs.update({f"video.{k}": np.array(v) for k, v in data_point.images.items()})
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text
        observations.append(parse_observation_gr00t(obs, loader.modality_configs))
    return observations


def stack_observations(observations: list[dict]) -> dict:
    """Concatenate B=1 observations into one batch."""
    batched = {}
    for modality in observations[0]:
        batched[modality] = {}
        for key, value in observations[0][modality].items():
            values = [o[modality][key] for o in observations]
            if isinstance(value, np.ndarray):
                batched[modality][key] = np.concatenate(values, axis=0)
            else:
                batched[modality][key] = [v for vs in values for v in vs]
    return batched


def build_model_inputs(
    policy: Gr00tPolicy, observation: dict, dtype=torch.bfloat16, return_images: bool = False
):
    """Steps 1-3 of ``Gr00tPolicy._get_action`` plus ``model.prepare_input``.

    With ``return_images`` also returns, per sample, the processed (C, H, W) uint8 frames the
    VLM actually sees, in token order.
    """
    processed = []
    for obs in policy._unbatch_observation(observation):
        vla_step_data = policy._to_vla_step_data(obs)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed.append(policy.processor(messages))
    collated = _rec_to_dtype(policy.collate_fn(processed), dtype=dtype)
    inputs = policy.model.prepare_input(collated["inputs"])
    if return_images:
        images = [[img.cpu() for img in p["vlm_content"]["images"]] for p in processed]
        return inputs, images
    return inputs
