# Baselines

Baselines for GR00T N1.7, each in its own section. All commands run from the repo root.

| Method | Code here | Paper | Official code |
|---|---|---|---|
| VLA-ADP | `gr00t_adp/` | [arXiv 2509.22093](https://arxiv.org/abs/2509.22093) | [chen7086/VLA-ADP](https://github.com/chen7086/VLA-ADP) |
| EfficientVLA | `gr00t_efficientvla/` | [arXiv 2506.10100](https://arxiv.org/abs/2506.10100) | none; implemented from the paper |
| VLA-Cache | `gr00t_vlacache/` | [arXiv 2502.02175](https://arxiv.org/abs/2502.02175) | [siyuhsu/vla-cache](https://github.com/siyuhsu/vla-cache) |
| GridS (Grid Sampler) | `gr00t_grids/` | [arXiv 2605.11817](https://arxiv.org/abs/2605.11817) | [Fediory/Grid-Sampler](https://github.com/Fediory/Grid-Sampler) |
| VLA-Pruner | `gr00t_vlapruner/` | [arXiv 2511.16449](https://arxiv.org/abs/2511.16449) | [MINT-SJTU/VLA-Pruner](https://github.com/MINT-SJTU/VLA-Pruner) |

## ADP visual token pruning (`gr00t_adp/`)

Paper: https://arxiv.org/abs/2509.22093 · Official code: [chen7086/VLA-ADP](https://github.com/chen7086/VLA-ADP)

ADP-style visual token pruning for GR00T N1.7. Image tokens are scored once, then the
low-scoring ones are dropped **before** the LLM.

**Score:** layer-0 Q·K logit from the task-text tokens to each image token (QK-norm, GQA).
No RoPE or softmax. Averaged over heads and text tokens.

### Configurations

| Name | What it does |
|---|---|
| `original` | Unmodified GR00T N1.7. Keeps all image tokens. |
| `keep=1.0` | ADP code path, no pruning. Output is identical to `original`; this is a sanity check. |
| `keep=r`, `per_view_uniform` | Keeps round(r · tokens) in **each** view. This is the default. |
| `keep=r`, `global` | Keeps the top r · total tokens across all views. Each view keeps at least `min_per_view`. |
| `keep=r`, `per_view_weighted` | Splits the total budget across views by `view_weights` (ADP's main/wrist split). |

Options (`ADPConfig`):
- `keep_ratio`: fraction of image tokens to keep (0, 1]
- `budget_policy`: one of the three policies above
- `view_weights`: `{view_key: weight}`, used by `per_view_weighted`
- `min_per_view`: minimum tokens kept per view (default 1)
- `text_source`: `task` (default, the instruction only) or `all_non_image`

### Commands

Set these first. GPU ids follow `nvidia-smi` order.

```bash
CKPT=<path/to/checkpoint>        # e.g. .../StopRollingCanTable_L3_cam5r_.../
DATASET=<path/to/lerobot/dataset>
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

**Unit tests** (CPU, no checkpoint needed)
```bash
uv run --no-sync --with pytest python -m pytest gr00t_adp/tests/test_token_scoring.py -q
```

**Equivalence check:** ADP matches the original at keep=1.0, and dropping tokens matches masking them out. CPU fp32.
```bash
CUDA_VISIBLE_DEVICES="" python -m gr00t_adp.tests.check_equivalence \
    --model-path $CKPT --dataset-path $DATASET --keep-ratio 0.5
```

**Score heatmaps:** one PNG per step. The top row is the score per view; the bottom row shows the kept tokens.
```bash
python -m gr00t_adp.visualize_scores --model-path $CKPT --dataset-path $DATASET \
    --steps 0 40 80 120 --keep-ratio 0.5 --budget-policy per_view_uniform --out-dir <out_dir>
```

**Latency:** compares `original` with each keep ratio. Reports prep / llm / head / model / policy times in ms.
```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m gr00t_adp.benchmark_latency \
    --model-path $CKPT --dataset-path $DATASET --device cuda \
    --keep-ratios 1.0 0.75 0.5 0.25 --batch-size 1 --output-json <out.json>
```

### Results so far (StopRollingCanTable, batch 1, A6000, bf16, FA2)

Model latency (ms) is about the same (~150) for 1, 3 and 5 views, and pruning gives **no
speedup**: keep=0.5 runs 0.88–0.96× as fast as `original`. At batch 1 the run is dominated by
kernel-launch overhead, and the action head takes ~95 ms of the ~150. On hold.

## EfficientVLA (`gr00t_efficientvla/`, arXiv 2506.10100)

Paper: https://arxiv.org/abs/2506.10100 · Official code: none (implemented from the paper)

Training-free. Three components, each of which can be switched off on its own:

1. **LLM layer pruning**: skip the layers whose input and output are most similar,
   `I(ℓ) = 1 − cos(x_ℓ, x_ℓ+1)`, measured on calibration samples. DeepStack features are
   still added at their original layer index.
2. **Visual token pruning inside the LLM**: run the first `prune_after` layers on all tokens.
   Score each image token by text→image attention in the last of those layers. In each view,
   keep the top `K_key` tokens, then fill `α` of the rest by score and the remainder by cosine
   distance to the key tokens.
3. **DiT feature caching**: recompute each DiT block's attention / MLP outputs every
   `cache_interval` denoising steps and reuse them in between (GR00T uses 4 steps).

Not included: the paper's 25% MLP weight compression (PruneNet).

### Configurations (`EfficientVLAConfig`)

| Option | Default | Meaning |
|---|---|---|
| `drop_layers` | `()` | LLM layer indices to skip; pick them with `drop_layers_from_importance(json, n)` (suggested n=4 of 16) |
| `keep_ratio` | 0.5 | Fraction of image tokens kept per view (`per_view_uniform`); 1.0 = off |
| `prune_after` | 2 | Layers run on all tokens before pruning |
| `num_key_tokens` | 4 | `K_key` per view |
| `alpha` | 0.5 | Score-based share of the remaining budget (rest is diversity) |
| `cache_interval` | 2 | DiT recompute interval; 1 = off |

`EfficientVLAConfig.disabled()` turns everything off and reproduces the original model exactly.

### Commands

**Unit tests**
```bash
uv run --no-sync --with pytest python -m pytest gr00t_efficientvla/tests/test_token_selection.py -q
```

**Equivalence check** (CPU fp32): all off == original; scores == real attention weights;
layer drop == identity layers; token drop == masking; cache == hook-based reference.
```bash
CUDA_VISIBLE_DEVICES="" python -m gr00t_efficientvla.tests.check_equivalence \
    --model-path $CKPT --dataset-path $DATASET
```

**Layer importance calibration**: writes the per-layer `I(ℓ)` and the drop ranking.
```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m gr00t_efficientvla.layer_importance \
    --model-path $CKPT --dataset-path $DATASET --output-json <out_dir>/layer_importance.json
```

**Latency ablation**: compares 8 configurations, every on/off combination of
L (layers), T (tokens) and C (cache). Reports backbone / head / model ms, speedup, and how far
action_pred moves from the original.
```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m gr00t_efficientvla.benchmark_latency \
    --model-path $CKPT --dataset-path $DATASET --importance-json <out_dir>/layer_importance.json \
    --num-drop-layers 4 --keep-ratio 0.5 --cache-interval 2 --output-json <out.json>
```

**Use as a policy** (a drop-in replacement for `Gr00tPolicy`)
```python
from gr00t_efficientvla import EfficientVLAConfig, Gr00tEfficientVLAPolicy, drop_layers_from_importance

config = EfficientVLAConfig(
    drop_layers=drop_layers_from_importance("<out_dir>/layer_importance.json", 4),
    keep_ratio=0.5,
    cache_interval=2,
)
policy = Gr00tEfficientVLAPolicy("new_embodiment", "<path/to/checkpoint>", device="cuda", evla_config=config)
```

### Results so far

The implementation passes every equivalence check for cam1 / cam3r / cam5r (B=1 and padded B=2).

Latency was measured on StopRollingCanTable cam5r (batch 1, A6000, bf16, FA2) with drop_layers
`(3, 4, 5, 6)` from calibration, keep 0.5 and cache interval 2. Model latency, mean of 50 runs:

| Config | backbone | head | model | speedup |
|---|---|---|---|---|
| original | 54 | 96 | 149 ms | 1.00× |
| T (tokens) | 61 | 97 | 159 | 0.94× |
| C (DiT cache) | 54 | 53 | 108 | **1.38×** |
| L (4 layers) | 49 | 97 | 144 | 1.03× |
| L+C | 50 | 55 | 104 | **1.43×** |
| L+T+C | 55 | 54 | 109 | 1.36× |

cam3r gives the same picture (L+T+C 1.44×).

- DiT caching gives almost all of the gain.
- Token pruning adds selection overhead and saves nothing at batch 1.
- Calibration ranks the layers similarly for every camera setup: layers 3–6 are least
  important, and layers 0–1 are by far the most important.
- Success rates have not been measured yet.

## VLA-Cache (`gr00t_vlacache/`, arXiv 2502.02175)

Paper: https://arxiv.org/abs/2502.02175 · Official code: [siyuhsu/vla-cache](https://github.com/siyuhsu/vla-cache)

Training-free. Reuses work across consecutive inferences of one episode (batch size 1):

1. **Static tokens**: in each view, tokens whose raw pixels have cosine similarity ≥ 0.996 with
   the frame of the previous inference. Up to 150/256 of a view's tokens.
2. **Minus task-relevant tokens**: drop the tokens with the most text→image attention in the
   previous inference (layer 7, top 100/256 per view).
3. **Reuse per layer**: at layers 2, 6, 9 and 11, a prefix of these tokens stops being
   computed. Its size comes from 1 − normalized attention entropy of the previous inference.
   Their K/V come from the cache.
4. GR00T-specific: the action head reads every token, so reused tokens also take their
   last-layer hidden state from the cache.

Differences from the reference OpenVLA-OFT code:

- The reference compares the current frame with itself (it reads the frame back right after
  appending it). Here the comparison is with the previous inference's frame.
- Text positions come from the token ids instead of hard-coded indices.
- The time spent on selection and attention statistics counts towards latency.

### Configurations (`VLACacheConfig`)

| Option | Default | Meaning |
|---|---|---|
| `enabled` | True | False = original model, no cache |
| `sim_threshold` | 0.996 | Pixel cosine threshold for a static token |
| `static_ratio` | 150/256 | Max share of a view's tokens that are static |
| `relevant_ratio` | 100/256 | Share of a view's tokens excluded as task-relevant |
| `relevance_layer` | 7 | Layer whose text→image attention gives relevance |
| `reuse_layers` | (2, 6, 9, 11) | Layers where reused tokens leave the queries |
| `growth_factor` | 0.55 | Damping of upward steps in the per-layer reuse schedule |

Call `policy.reset()` at every episode start. A changed prompt or image layout rebuilds the cache.

### Commands

**Unit tests**
```bash
uv run --no-sync --with pytest python -m pytest gr00t_vlacache/tests/test_static_tokens.py -q
```

**Equivalence check** (CPU fp32): disabled / first inference == original; entropy and relevance
== eager attention; same frame twice == full computation; frame A→B == a reference that runs
all tokens through the original layers and swaps in only the cached K/V.
```bash
CUDA_VISIBLE_DEVICES="" python -m gr00t_vlacache.tests.check_equivalence \
    --model-path $CKPT --dataset-path $DATASET --steps 40 56
```

**Use as a policy**
```python
from gr00t_vlacache import Gr00tVLACachePolicy, VLACacheConfig

policy = Gr00tVLACachePolicy("new_embodiment", "<path/to/checkpoint>", device="cuda",
                             vlac_config=VLACacheConfig())
policy.reset()  # at every episode start
```

### Results so far

All equivalence checks pass for cam1 / cam3r / cam5r.

On trajectory 0, steps 40 → 56 (cam5r):
- 38/64 tokens per view are static.
- 128 of 320 image tokens are reusable, but the entropy schedule reuses only 1 at layer 2 and
  83 from layer 6 on.
- Features differ from a full recompute by 9e-4 (relative).

Latency and success rates have not been measured yet.

## Grid Sampler (`gr00t_grids/`, GridS, arXiv 2605.11817)

Paper: https://arxiv.org/abs/2605.11817 · Official code: [Fediory/Grid-Sampler](https://github.com/Fediory/Grid-Sampler)

**Needs training.** A small learned module sits between the vision merger and the LLM and
reads K tokens per image:

- mean-pooled token map → MLP → sigmoid → K coordinates
- the tokens are read at those coordinates by bilinear interpolation
- a coordinate embedding is added to each sampled token
- the LLM and the action head see only K tokens per image
- the DeepStack maps are read at the same coordinates

It is trained from the base model with the usual run configuration (same tune flags, steps,
batch size and lr). The sampler is randomly initialized and learns only from the action loss,
through the frozen LLM. As in every Grid-Sampler implementation, one sampler is shared by all
views by default.

### Configurations (`GridSConfig`, `--grids-*` flags)

| Option | Default | Meaning |
|---|---|---|
| `num_tokens` | 16 | K, tokens per image (an image has 64 at 256 px) |
| `position_mode` | `continuous` | MRoPE of sampled tokens: `continuous` (h, w at the predicted sub-token coordinate), `grid` (√K×√K grid), `first` (first K positions) |
| `embed_coords` | True | Add the coordinate MLP embedding |
| `sample_deepstack` | True | Read the DeepStack maps at the same coordinates |
| `share_across_views` | True | One sampler for all views (False: one per view, set `num_views`) |
| `input_layernorm` | False | LayerNorm before the coordinate MLP (`grid_new.py` variant) |

The config is saved to `experiment_cfg/grids_config.json`, which is copied into every checkpoint.

### Commands

**Train**: same arguments as `examples/finetune.sh`; GridS flags go after `--`.
```bash
NUM_GPUS=<n> MAX_STEPS=<steps> GLOBAL_BATCH_SIZE=<bs> bash gr00t_grids/finetune.sh \
    --base-model-path <base model> --dataset-path $DATASET \
    --embodiment-tag <tag> --output-dir <out_dir> \
    -- --grids-num-tokens 16 --grids-position-mode continuous
```

**Unit tests**
```bash
uv run --no-sync --with pytest python -m pytest gr00t_grids/tests/test_sampler.py -q
```

**Integration check** (CPU fp32): identity sampling (K = 64, grid-point coordinates) ==
original; K-token sequences, positions and padded batches; gradients reach the sampler and
not the frozen LLM / ViT; save → `Gr00tGridSPolicy` load round trip.
```bash
CUDA_VISIBLE_DEVICES="" python -m gr00t_grids.tests.check_equivalence \
    --model-path $CKPT --dataset-path $DATASET --work-dir <scratch_dir> --num-tokens 16
```

**Use as a policy** (checkpoints trained with `gr00t_grids`)
```python
from gr00t_grids import Gr00tGridSPolicy

policy = Gr00tGridSPolicy("<embodiment tag>", "<path/to/grids/checkpoint>", device="cuda")
```

### Results so far

All integration checks pass on two existing checkpoints:

- cam5r: 5 views, sequence 344 → 104 at K=16
- cam1: 1 view, sequence 80 → 32

No GridS model has been trained yet. The target is the external robocasa-gr1-tabletop-tasks
run configuration.

## VLA-Pruner (`gr00t_vlapruner/`, arXiv 2511.16449)

Paper: https://arxiv.org/abs/2511.16449 · Official code: [MINT-SJTU/VLA-Pruner](https://github.com/MINT-SJTU/VLA-Pruner)

Training-free; batch size 1. Prunes image tokens inside the LLM after layer K = 3 using two
signals:

- **Semantic**: layer-K attention on each image token, averaged over heads and over all valid
  query rows.
- **Action**: the DiT's action-token → image-token cross-attention from previous inferences.
  It is averaged over the latter-half image cross-attention blocks (18, 22, 26, 30) and all 4
  flow-matching steps, then smoothed over the last 3 inferences with decay 0.8. Pruned tokens
  score 0.
- **Combine-then-Filter**, per view: take top-k by semantic ∪ top-k by action. If that is
  larger than k, keep k by greedy max-min cosine distance on the tokens' LLM input embeddings.
- The first 3 inferences of an episode run unpruned while the history fills.

This follows the reference OpenVLA-OFT code (`_vlapruner_image_indices`,
`_redundancy_minization`, `_build_historical_action_attention`). For π0 the paper averages over
flow-matching steps; the DiT stands in for OFT's in-LLM action tokens.

### Configurations (`VLAPrunerConfig`)

| Option | Default | Meaning |
|---|---|---|
| `keep_ratio` | 0.25 | Image tokens kept per view (paper: 0.5 / 0.25 / 0.125) |
| `prune_layer` | 3 | Score with layer K's attention and drop after it (layers 0..K see all tokens) |
| `history_window` / `history_decay` | 3 / 0.8 | Temporal smoothing of the action relevance |
| `action_blocks` | `latter_half` | DiT image cross-attention blocks used (`all` = all 8) |
| `action_steps` | `all` | Denoising steps used (`latter_half` = steps 2, 3) |

Call `policy.reset()` at every episode start.

### Commands

**Unit tests**: checked against a verbatim port of the reference selection code.
```bash
uv run --no-sync --with pytest python -m pytest gr00t_vlapruner/tests/test_selection.py -q
```

**Equivalence check** (CPU fp32, consecutive frames). It checks that:
- disabled and history-filling inferences == original;
- semantic scores == eager layer-K attention;
- the recomputed DiT attention reproduces the real block outputs;
- dropping tokens == masking them;
- history zeros sit exactly at the pruned tokens.
```bash
CUDA_VISIBLE_DEVICES="" python -m gr00t_vlapruner.tests.check_equivalence \
    --model-path $CKPT --dataset-path $DATASET --keep-ratio 0.25
```

**Use as a policy**
```python
from gr00t_vlapruner import Gr00tVLAPrunerPolicy, VLAPrunerConfig

policy = Gr00tVLAPrunerPolicy("new_embodiment", "<path/to/checkpoint>", device="cuda",
                              vlap_config=VLAPrunerConfig(keep_ratio=0.25))
policy.reset()  # at every episode start
```

### Results so far

All equivalence checks pass for cam1 / cam3r / cam5r. At keep 0.25: 16/64 tokens per view, sequence 344 → 104 (cam5r);
the semantic ∪ action union exceeded the budget in every view, so the redundancy filter ran.

Latency and success rates have not been measured yet.

## Closed-loop evaluation on LIBERO (`gr00t_baselines_eval/`)

Success rate of the original model and each baseline on the public fine-tuned
`nvidia/GR00T-N1.7-LIBERO` checkpoint. It uses the repo's LIBERO env and wrappers, with three
changes from `gr00t/eval/rollout_policy.py`:

- one env at a time (batch size 1, which VLA-Cache and VLA-Pruner need);
- `policy.reset()` after every episode;
- the same seed per task for every method.

The server is a normal GR00T policy server plus a `stats` endpoint: method, config and
`get_action` time.

### Commands

Set up the LIBERO env first, as in `examples/LIBERO/README.md`.

```bash
CKPT=<path/to/GR00T-N1.7-LIBERO/libero_10>
export TMPDIR=$HOME/tmp    # robosuite also writes /tmp/robosuite.log unless FILE_LOGGING_LEVEL=None

# 1) Policy server (model venv). --config is a JSON of the method's config fields.
CUDA_VISIBLE_DEVICES=<gpu> python -m gr00t_baselines_eval.server \
    --model-path $CKPT --embodiment-tag LIBERO_PANDA --port 5555 \
    --method vlapruner --config '{"keep_ratio": 0.5}'

# 2) Rollout (LIBERO venv). Writes after each task; a rerun skips finished tasks.
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python -m gr00t_baselines_eval.rollout \
    --suite libero_10 --tasks 0 1 2 3 4 --n-episodes 50 --port 5555 \
    --output-json <out_dir>/vlapruner_keep0.5.json
```

`--method` is one of `original`, `adp`, `efficientvla`, `vlacache` or `vlapruner`. For
EfficientVLA, `{"importance_json": <file>, "num_drop_layers": n}` can replace `drop_layers`.
Calibrate with `gr00t_efficientvla.layer_importance` on LIBERO-10 data, for example 20 episodes
of `IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot`.

### Results: LIBERO-10, first 5 tasks, 50 episodes each, seed 0

The checkpoint has 2 views (`image`, `wrist_image`) of 64 tokens each. The published success
rate on all 10 tasks is 94.35%.

| Method | Config | Successes per task (/50) | Mean |
|---|---|---|---|
| original | — | 48, 49, 47, 45, 44 | **93.2%** |
| ADP | keep 0.75, per_view_uniform | 47, 49, 46, 48, 42 | 92.8% |
| VLA-Cache | defaults | 47, 50, 47, 47, 42 | **93.2%** |
| VLA-Pruner | keep 0.25 (default) | 23, 30, 49, 45, 30 | 70.8% |
| VLA-Pruner | keep 0.5 | 43, 48, 44, 49, 40 | 89.6% |
| EfficientVLA | drop (3,4,5,6), keep 0.5, cache 2 | 0 (task 0, stopped) | 0% |
| EfficientVLA | drop (3,5), keep 0.75, cache 2 | 40, 42, 45, 49, 45 | 88.4% |

EfficientVLA ablation, using the full config's components one at a time (task 0, 10 episodes):

| Component | Successes (/10) |
|---|---|
| layers only | 7 |
| tokens only | 6 |
| DiT cache only | 10 |

The equivalence checks pass on this checkpoint, so the drop is not an implementation bug.

Findings:
- ADP and VLA-Cache match the original model.
- VLA-Pruner at keep 0.25 fails mostly on tasks 0, 1 and 4, the long two-object tasks.
- EfficientVLA's loss comes from layer dropping and token pruning, not from caching.

The rollout's per-call times (120–350 ms) are not compared. Up to 6 runs shared one GPU on a
loaded CPU. Latency was measured separately, as below.

### Latency and token counts: same checkpoint, batch 1, A6000 alone, bf16

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m gr00t_baselines_eval.benchmark_latency \
    --model-path $CKPT --dataset-path <libero_10 lerobot dataset> --output-json <out.json> \
    --traj-ids 0 1 --max-inferences 40 --repeats 2 \
    --runs 'original|original|{}' 'vlapruner_0.5|vlapruner|{"keep_ratio": 0.5}'   # name|method|config
```

How it was measured:
- Two dataset trajectories are replayed as episodes: one inference every 8 steps,
  `policy.reset()` at each start. This lets VLA-Cache and VLA-Pruner use their history.
- The first 3 inferences of each episode are not counted, which leaves 114 inferences per row.
- Token counts are read with hooks:
  - image tokens each LLM layer actually computes (0 for a skipped layer);
  - image tokens the DiT head reads.
- The input is 128 image tokens (2 views × 64) plus 28 text tokens. The LLM has 16 layers.

| Run | Policy ms | Model ms | LLM | DiT head | Speedup | Image tokens per LLM layer | Token-layers | DiT image tokens |
|---|---|---|---|---|---|---|---|---|
| original | 155.9 | 150.9 | 53.6 | 97.3 | 1.00× | 128 ×16 | 100% | 128 |
| ADP keep 0.75 | 167.6 | 162.0 | 57.4 | 104.6 | 0.93× | 96 ×16 | 75% | 96 |
| VLA-Cache | 187.1 | 181.4 | 74.4 | 107.0 | 0.83× | 128 ×2, 127 ×4, 105 ×10 (reuse) | 89% | 128 |
| VLA-Pruner keep 0.25 | 168.1 | 162.9 | 58.8 | 104.0 | 0.93× | 128 ×4, 32 ×12 | 44% | 32 |
| VLA-Pruner keep 0.5 | 172.9 | 167.7 | 63.4 | 104.4 | 0.90× | 128 ×4, 64 ×12 | 62% | 64 |
| EfficientVLA full (drop 3–6, keep 0.5, cache 2) | 113.9 | 107.7 | 49.8 | 57.9 | **1.40×** | 128 ×2, 64, 0 ×4, 64 ×9 | 44% | 64 |
| EfficientVLA mild (drop 3,5, keep 0.75, cache 2) | 116.5 | 110.9 | 53.0 | 57.9 | **1.36×** | 128 ×2, 96, 0, 96, 0, 96 ×10 | 69% | 96 |
| EfficientVLA cache only (cache 2) | 113.9 | 108.6 | 51.0 | 57.6 | **1.39×** | 128 ×16 | 100% | 128 |

Notes:
- Policy ms is the whole `get_action`. Model ms covers LLM + DiT head (vision encoder in LLM).
- Token-layers is the sum of image tokens over the 16 layers.

What this shows:
- At batch 1, fewer tokens do not mean less time. Cutting the LLM's image work to 44% saves
  nothing, because the GPU is mostly waiting on kernel launches, not doing arithmetic.
- The selection work each method adds makes it slower than the original:
  - top-k for ADP;
  - attention scores and the redundancy filter for VLA-Pruner;
  - entropy and relevance for every layer for VLA-Cache.
- Only DiT caching helps. Skipping 2 of 4 denoising computations cuts the head from 97 to
  58 ms.
- EfficientVLA cache-only gets the same 1.39× as the full config. In the 10-episode ablation
  above it kept 10/10 successes, so the speedup does not cost success rate.

### Keep-ratio sweep: token pruning methods, LIBERO-10 first 5 tasks

Every run uses 25 episodes per task. Runs that had 50 episodes are cut to their first 25;
the seed is the same, so the initial states match. Latency is batch-1 model time relative to
`original`, measured in two sessions with about ±5% noise. "Token-layers" is the image tokens
computed summed over the 16 LLM layers. The DiT column is the image tokens the DiT reads.
"EfficientVLA tokens" runs token pruning only: no layer drop, `cache_interval=1`.

| Method | keep | Successes per task (/25) | Success | Speed | Token-layers | DiT tokens |
|---|---|---|---|---|---|---|
| original | — | 24, 24, 23, 23, 21 | **92.0%** | 1.00× | 100% | 128 |
| ADP | 0.75 | 23, 24, 24, 24, 21 | 92.8% | 0.93× | 75% | 96 |
| ADP | 0.5 | 23, 25, 24, 25, 22 | 95.2% | 1.02× | 50% | 64 |
| ADP | 0.25 | 10, 9, 13, 22, 2 | 44.8% | 1.01× | 25% | 32 |
| ADP | 0.125 | 0, 0, 0, 1, 0 | 0.8% | 1.01× | 12% | 16 |
| VLA-Pruner | 0.75 | 23, 24, 24, 25, 24 | 96.0% | 0.86× | 81% | 96 |
| VLA-Pruner | 0.5 | 24, 24, 23, 25, 20 | 92.8% | 0.90× | 62% | 64 |
| VLA-Pruner | 0.25 | 11, 15, 24, 24, 16 | 72.0% | 0.93× | 44% | 32 |
| VLA-Pruner | 0.125 | 4, 9, 13, 14, 1 | 32.8% | 0.87× | 34% | 16 |
| EfficientVLA tokens | 0.75 | 23, 20, 25, 24, 23 | 92.0% | 0.93× | 78% | 96 |
| EfficientVLA tokens | 0.5 | 17, 10, 20, 23, 16 | 68.8% | 0.97× | 56% | 64 |
| EfficientVLA tokens | 0.25 | 0, 0, 6, 0, 0 | 4.8% | 1.02× | 34% | 32 |
| EfficientVLA tokens | 0.125 | 0, 0, 0, 0, 0 | 0.0% | 1.01× | 23% | 16 |
| VLA-Cache | defaults | 25, 25, 25, 23, 21 | 95.2% | 0.83× | 89% | 128 |
| EfficientVLA mild | drop 3,5 + keep 0.75 + cache 2 | 21, 21, 22, 24, 21 | 87.2% | **1.36×** | 69% | 96 |

Findings:
- At keep ≥ 0.5, ADP and VLA-Pruner match the original. EfficientVLA tokens matches only at
  0.75.
- Below that the methods separate. VLA-Pruner degrades most gracefully, likely because it
  also uses the DiT's action attention:

  | keep | VLA-Pruner | ADP | EfficientVLA tokens |
  |---|---|---|---|
  | 0.25 | 72% | 45% | 5% |
  | 0.125 | 33% | 1% | 0% |

- EfficientVLA's single-layer (layer 1) score with diversity filling is the weakest selector here.
- No token-pruning setting is faster at batch 1. Only DiT caching (EfficientVLA) is.

Commands: `gr00t_baselines_eval.benchmark_latency` and `gr00t_baselines_eval.rollout` as
above, with `--n-episodes 25`. The EfficientVLA tokens config is
`{"keep_ratio": r, "cache_interval": 1}`.
