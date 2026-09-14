"""
``fastcellstates`` command line.

    fastcellstates data.h5ad -o out/                       # `fast` preset
    fastcellstates data.h5ad --preset exact -o out/
    fastcellstates data.h5ad --cfg.model.theta 4096        # one-field override
    fastcellstates data.h5ad --print_config > run.yaml     # save the effective config
    fastcellstates data.h5ad --config run.yaml             # ... and replay it

``--preset`` seeds the config; ``--cfg.<block>.<field>`` and ``--config`` layer
on top; ``--print_config`` dumps the fully-resolved tree as YAML.
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import PRESETS, Config
from .pipeline import run


def _make_parser(cfg_default: Config):
    from jsonargparse import ArgumentParser, set_parsing_settings

    # Config's fields are documented with attribute docstrings (a bare string
    # literal right after each field, IDE/Sphinx-recognised); this makes
    # jsonargparse pick those up for --help instead of leaving every field at
    # a bare "(type: ..., default: ...)".
    set_parsing_settings(docstring_parse_attribute_docstrings=True)

    p = ArgumentParser(
        prog="fastcellstates", description="Dirichlet-multinomial cell-state clustering."
    )
    p.add_argument("data", nargs="+", help="UMI file(s): .h5ad / .mtx / .tsv / .csv / .npy")
    p.add_argument("-o", "--out", default=None, help="dir for summary.npz + labels.txt")
    p.add_argument(
        "--preset", choices=list(PRESETS), default="fast", help="starting config (--cfg.* override)"
    )
    p.add_argument("--config", action="config", help="load a YAML config")
    p.add_class_arguments(Config, "cfg", default=asdict(cfg_default))
    return p


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # stage 1 (stdlib argparse; jsonargparse forbids parse_known_args): which
    # preset?  It seeds the `cfg` default; the real parser validates the choice.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default="fast")
    preset = pre.parse_known_args(argv)[0].preset

    parser = _make_parser(PRESETS.get(preset, PRESETS["fast"]))
    ns = parser.instantiate(parser.parse_args(argv))
    cfg: Config = ns["cfg"]

    data = ns["data"] if len(ns["data"]) > 1 else ns["data"][0]
    summ = run(data, cfg)
    print(f"{summ.n_states} states / {summ.n_cells} cells | Theta {summ.theta:g}")
    if ns["out"]:
        outp = Path(ns["out"])
        outp.mkdir(parents=True, exist_ok=True)
        summ.save(outp / "summary.npz")
        np.savetxt(outp / "labels.txt", summ.labels, fmt="%d")
        print(f"wrote {outp}/summary.npz, {outp}/labels.txt")
