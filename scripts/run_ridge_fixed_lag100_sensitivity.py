"""Run the fixed 100-lag Ridge B3 sensitivity. Does not train neural models."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.fixed_lag100_sensitivity import run_fixed_lag100_sensitivity  # noqa: E402


def main() -> int:
    ctx = run_fixed_lag100_sensitivity(ROOT)
    print(f"Wrote {ctx['out']}", flush=True)
    for row in ctx["summary"]:
        p = ctx["primary"][row["evaluation_context"]]
        delta = float(row["mean_sequence_rmse"]) - float(p["mean_sequence_rmse"])
        print(
            f"{row['evaluation_context']}: fixed100={float(row['mean_sequence_rmse']):.6f} "
            f"primary_B3={float(p['mean_sequence_rmse']):.6f} delta={delta:+.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
