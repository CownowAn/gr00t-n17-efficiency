#!/usr/bin/env bash
# Same interface as examples/finetune.sh; GridS options go after "--", e.g.
#   bash gr00t_grids/finetune.sh --base-model-path ... --dataset-path ... --embodiment-tag ... \
#       --output-dir ... -- --grids-num-tokens 16
# examples/finetune.sh is reused as is, with only its launcher path pointed at gr00t_grids.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bash <(sed 's#gr00t/experiment/launch_finetune.py#gr00t_grids/launch_finetune.py#' examples/finetune.sh) "$@"
