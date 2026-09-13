"""Rebuild publication_current/ from frozen Stage-2/3 CSVs.

Does not train models, retune hyperparameters, or fit target-domain objects.
Never selects a Ridge comparator by looking at target-domain RMSE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage4.publication_current import (  # noqa: E402
    FIXED100_FILES,
    PRIMARY_RIDGE_ID,
    RIDGE_NAME,
    run_publication_current,
)

B3_UZH_TO_EUROC = 0.83080377215
B2_UZH_TO_EUROC = 0.8153206077
B2_ROUNDED = 0.815321


def _fail(errors: list[str], msg: str) -> None:
    errors.append(msg)


def verify_publication_package(root: Path, mean: float) -> list[str]:
    errors: list[str] = []
    pub = root / "publication_current"
    src = (root / "src" / "stage4" / "publication_current.py").read_text(encoding="utf-8")

    if abs(mean - B2_UZH_TO_EUROC) < 1e-6 or abs(mean - B2_ROUNDED) < 5e-7:
        _fail(errors, f"UZH->EuRoC primary Ridge is the old B2 value ({mean:.12f}).")
    if abs(mean - B3_UZH_TO_EUROC) > 1e-8:
        _fail(errors, f"UZH->EuRoC primary Ridge {mean:.12f} is not source-selected B3 {B3_UZH_TO_EUROC}.")
    if PRIMARY_RIDGE_ID != "B3_RIDGE_QF_QW":
        _fail(errors, f"PRIMARY_RIDGE_ID is {PRIMARY_RIDGE_ID}, not B3_RIDGE_QF_QW.")

    spec_path = pub / "FILTER_SPECIFICATION_CORRECTED.md"
    if not spec_path.exists():
        _fail(errors, "missing publication_current/FILTER_SPECIFICATION_CORRECTED.md")
    else:
        spec = spec_path.read_text(encoding="utf-8")
        if "## EuRoC design" not in spec or "## UZH design" not in spec:
            _fail(errors, "FILTER_SPECIFICATION_CORRECTED.md missing EuRoC/UZH headings")
        else:
            euroc = spec.split("## EuRoC design", 1)[1].split("## UZH design", 1)[0]
            uzh = spec.split("## UZH design", 1)[1]
            if "order: 6" in euroc:
                _fail(errors, "EuRoC filter is labelled order 6 in FILTER_SPECIFICATION_CORRECTED.md")
            if "order: 5" not in euroc:
                _fail(errors, "EuRoC filter is not labelled order 5 in FILTER_SPECIFICATION_CORRECTED.md")
            if "order: 6" not in uzh:
                _fail(errors, "UZH-FPV filter is not labelled order 6 in FILTER_SPECIFICATION_CORRECTED.md")

    audit = pub / "RESAMPLING_AUDIT_CORRECTED.csv"
    if not audit.exists():
        _fail(errors, "missing publication_current/RESAMPLING_AUDIT_CORRECTED.csv")
    else:
        audit_txt = audit.read_text(encoding="utf-8")
        if "ellip_sos_order6_fs200" in audit_txt:
            _fail(errors, "EuRoC filter is labelled order 6 in RESAMPLING_AUDIT_CORRECTED.csv")
        if "ellip_sos_order5_fs200" not in audit_txt:
            _fail(errors, "EuRoC filter_name order 5 missing from RESAMPLING_AUDIT_CORRECTED.csv")
        if "ellip_sos_order6_fs500" not in audit_txt:
            _fail(errors, "UZH-FPV filter_name order 6 missing from RESAMPLING_AUDIT_CORRECTED.csv")

    for name in FIXED100_FILES:
        path = pub / name
        if not path.exists() or path.stat().st_size == 0:
            _fail(errors, f"fixed100 sensitivity file missing: {path}")
        alt = root / "results" / "sensitivity" / name
        if not alt.exists():
            _fail(errors, f"fixed100 sensitivity file missing: {alt}")

    if "np.argmin" in src:
        _fail(errors, "target-selected Ridge logic detected: np.argmin in publication_current.py")
    if "BEST_RIDGE_ID" in src:
        _fail(errors, "target-selected Ridge logic detected: BEST_RIDGE_ID in publication_current.py")
    if 'PRIMARY_RIDGE_ID = "B3_RIDGE_QF_QW"' not in src:
        _fail(errors, "publication_current.py does not hardcode PRIMARY_RIDGE_ID = B3_RIDGE_QF_QW")
    if "never inspects target-domain RMSE to choose a Ridge variant" not in src:
        _fail(errors, "publication_current.py missing source-only Ridge selection statement")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild manuscript-facing publication_current/ from frozen CSVs.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Fail if B2 Ridge, EuRoC order-6 labels, missing fixed100 files, or target-selected Ridge logic are present.",
    )
    args = parser.parse_args()

    ctx = run_publication_current(ROOT)
    mean = float(ctx["uzh_to_euroc_ridge_mean"])
    print("NO NEW MODEL WAS TRAINED.")
    print("NO HYPERPARAMETER WAS RETUNED.")
    print("NO TARGET-DOMAIN ADAPTATION WAS PERFORMED.")
    print("PRIMARY RIDGE COMPARATOR = B3_RIDGE_QF_QW (source-only freeze, all four contexts).")
    print("FILTER ORDERS (publication-facing): EuRoC/200 Hz = 5; UZH-FPV/500 Hz = 6.")
    print("FIXED100 RIDGE is supplementary; primary lag remains source-selected.")
    print(f"UZH_TO_EUROC {RIDGE_NAME} mean RMSE = {mean:.12f}")
    print(f"UZH_TO_EUROC 50 ms = {float(ctx['uzh_to_euroc_ridge_50']):.12f}")
    print(f"UZH_TO_EUROC 100 ms = {float(ctx['uzh_to_euroc_ridge_100']):.12f}")
    print(f"UZH_TO_EUROC 200 ms = {float(ctx['uzh_to_euroc_ridge_200']):.12f}")
    print(f"Wrote {ctx['out']}")
    if abs(mean - B3_UZH_TO_EUROC) > 1e-8:
        print("ERROR: UZH_TO_EUROC primary Ridge mean is not the source-selected B3 value.")
        return 1
    if abs(mean - B2_UZH_TO_EUROC) < 1e-6:
        print("ERROR: UZH_TO_EUROC still using supplementary B2 Ridge.")
        return 1
    if args.verify:
        errors = verify_publication_package(ROOT, mean)
        if errors:
            print("VERIFY FAILED:")
            for item in errors:
                print(f"  - {item}")
            return 1
        print("VERIFY PASSED: B3 primary, filter orders 5/6, fixed100 present, no target-selected Ridge logic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
