"""Run Stage 0C EuRoC+UZH audit after official archives are present."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit.stage0c_run import run_stage0c  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Also run Phase A download")
    args = parser.parse_args()
    ctx = run_stage0c(ROOT, do_download=bool(args.download))
    print(f"STAGE0C_DECISION={ctx['decision']}")
    print(f"euroc_audited={ctx['n_euroc_audited']} euroc_primary={ctx['n_euroc_primary']} uzh_pass={ctx['n_uzh_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
