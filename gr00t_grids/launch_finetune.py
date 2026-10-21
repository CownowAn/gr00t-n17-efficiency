# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 finetuning with the Grid Sampler.

Takes every argument of ``gr00t/experiment/launch_finetune.py`` unchanged, plus ``--grids-*``
options. It installs the GridS model pipeline and then runs the original launcher, so the run
configuration (tune flags, steps, batch size, learning rate, ...) is exactly the usual one.

    python gr00t_grids/launch_finetune.py <launch_finetune.py args> --grids-num-tokens 16
"""

import argparse
from dataclasses import fields
from pathlib import Path
import runpy
import sys


ORIGINAL_LAUNCHER = (
    Path(__file__).resolve().parents[1] / "gr00t" / "experiment" / "launch_finetune.py"
)


def parse_grids_args(argv: list[str]):
    from gr00t_grids.config import GridSConfig

    parser = argparse.ArgumentParser(add_help=False)
    for f in fields(GridSConfig):
        flag = "--grids-" + f.name.replace("_", "-")
        if f.type in (bool, "bool"):
            parser.add_argument(flag, dest=f.name, type=lambda v: v.lower() in ("1", "true", "yes"))
        else:
            parser.add_argument(flag, dest=f.name, type=type(f.default))
    known, rest = parser.parse_known_args(argv)
    values = {k: v for k, v in vars(known).items() if v is not None}
    return GridSConfig(**values), rest


def main():
    repo_root = str(ORIGINAL_LAUNCHER.parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from gr00t_grids.pipeline import install

    config, rest = parse_grids_args(sys.argv[1:])
    install(config)
    print(f"[GridS] {config}")
    sys.argv = [str(ORIGINAL_LAUNCHER), *rest]
    runpy.run_path(str(ORIGINAL_LAUNCHER), run_name="__main__")


if __name__ == "__main__":
    main()
