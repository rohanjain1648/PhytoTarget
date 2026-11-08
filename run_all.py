"""Run the whole PhytoTarget pipeline in order.

    python run_all.py              # everything
    python run_all.py --skip-fetch # reuse already-downloaded data

Each stage is a module with its own CLI, so this is a convenience wrapper rather
than a place where logic lives. Stages are idempotent: rerunning one overwrites
its own outputs and touches nothing else.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Several parallel LightGBM fits each spawning one thread per core will exhaust a
# laptop and surface as an OpenBLAS allocation failure. Cap the pools before numpy
# is imported anywhere downstream.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from src.util import setup_logging  # noqa: E402

log = logging.getLogger("run_all")

STAGES = [
    ("fetch_chembl", "bioactivity labels from ChEMBL", True),
    ("fetch_lotus", "plant-compound library from LOTUS/Wikidata", True),
    ("augment_library", "guarantee retrospective compound coverage", True),
    ("train", "scaffold split, y-scramble, AVE, calibration", False),
    ("screen", "score the natural-product library", False),
    ("enrich", "hypergeometric organism enrichment", False),
    ("kg", "knowledge graph", False),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-fetch", action="store_true",
                    help="reuse data already in data/processed")
    ap.add_argument("--target", help="restrict to one ChEMBL target id")
    args = ap.parse_args(argv)

    setup_logging()
    import importlib

    for name, description, is_fetch in STAGES:
        if is_fetch and args.skip_fetch:
            log.info("skipping %s (--skip-fetch)", name)
            continue
        log.info("=" * 68)
        log.info("STAGE %-16s %s", name, description)
        log.info("=" * 68)
        module = importlib.import_module(f"src.{name}")
        argv_stage = []
        if args.target and name in {"train", "screen", "enrich", "kg", "fetch_chembl"}:
            argv_stage = ["--target", args.target]
        t0 = time.time()
        rc = module.main(argv_stage)
        if rc:
            log.error("stage %s failed with code %s - stopping", name, rc)
            return rc
        log.info("stage %s finished in %.1fs", name, time.time() - t0)

    log.info("pipeline complete. launch the dashboard with:")
    log.info("    streamlit run app/streamlit_app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
