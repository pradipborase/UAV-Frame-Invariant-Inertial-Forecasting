"""Verify official EuRoC sources and write Stage 0C source artefacts (no large download)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit.downloader import utc_now  # noqa: E402
from audit.stage0c import (  # noqa: E402
    REMOTE_INV_COLUMNS,
    load_stage0c_config,
    official_remote_inventory,
    results_dir,
    write_csv,
    write_source_verification,
)


def main() -> int:
    cfg = load_stage0c_config(ROOT)
    out = results_dir(ROOT)
    write_source_verification(ROOT, cfg, utc_now())
    write_csv(out / "EUROC_REMOTE_SEQUENCE_INVENTORY.csv", REMOTE_INV_COLUMNS, official_remote_inventory(cfg))
    print(f"Wrote {out / 'EUROC_SOURCE_VERIFICATION.md'}")
    print(f"Wrote {out / 'EUROC_REMOTE_SEQUENCE_INVENTORY.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
