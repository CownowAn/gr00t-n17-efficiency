# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field


SELECTIONS = ("all", "latter_half")


@dataclass
class VLAPrunerConfig:
    # Master switch; False runs the original model and records nothing.
    enabled: bool = True
    # Fraction of each view's image tokens kept (the reference scripts use 0.5 / 0.25 / 0.125).
    keep_ratio: float = 0.25
    # Tokens are scored with the attention of LLM layer K and dropped after it (fastv_k = 3).
    prune_layer: int = 3
    # Temporal smoothing of the action relevance: window w and decay gamma. Pruning starts once
    # w inferences of the episode have been recorded.
    history_window: int = 3
    history_decay: float = 0.8
    # Action relevance = DiT action-token -> image-token cross-attention of the previous
    # inferences, averaged over these image cross-attention blocks and denoising steps
    # (paper Sec. 4.3: latter half layers, averaged over flow-matching steps).
    action_blocks: str = "latter_half"
    action_steps: str = "all"
    # Video modality keys in processor order; set from the policy's modality config.
    view_keys: list[str] = field(default_factory=list)
    # Keep the per-inference scores and selections on the backbone for inspection.
    record: bool = False

    def __post_init__(self):
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.prune_layer < 0:
            raise ValueError("prune_layer must be >= 0")
        if self.history_window < 1:
            raise ValueError("history_window must be >= 1")
        if not 0.0 < self.history_decay <= 1.0:
            raise ValueError("history_decay must be in (0, 1]")
        for name in ("action_blocks", "action_steps"):
            if getattr(self, name) not in SELECTIONS:
                raise ValueError(f"{name} must be one of {SELECTIONS}")
