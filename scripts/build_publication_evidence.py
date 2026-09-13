"""Historical Stage-4 audit builder (pre-correction Ridge map).

Does not train models, retune hyperparameters, or fit target-domain objects.

Manuscript-facing outputs are produced by scripts/build_publication_current.py,
which uses source-selected B3 Ridge in all four contexts. This historical
builder retains the descriptive UZH→EuRoC B2 comparator and is not the
public manuscript generator.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage4.run import grab, run_stage4  # noqa: E402


def _patch_pytest(passed: str, failed: str) -> None:
    for path in (ROOT / "STAGE4_REPORT.md", ROOT / "results" / "stage4" / "STAGE4_REPORT.md"):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace("SEE_PYTEST passed", f"{passed} passed")
        text = text.replace("SEE_PYTEST failed", f"{failed} failed")
        path.write_text(text, encoding="utf-8")
    pub = ROOT / "publication_package" / "evidence" / "STAGE4_REPORT.md"
    if pub.exists():
        text = pub.read_text(encoding="utf-8")
        text = text.replace("SEE_PYTEST passed", f"{passed} passed")
        text = text.replace("SEE_PYTEST failed", f"{failed} failed")
        pub.write_text(text, encoding="utf-8")


def _parse_pytest(output: str) -> tuple[str, str]:
    passed = "0"
    failed = "0"
    m_pass = re.search(r"(\d+) passed", output)
    m_fail = re.search(r"(\d+) failed", output)
    if m_pass:
        passed = m_pass.group(1)
    if m_fail:
        failed = m_fail.group(1)
    return passed, failed


def _fmt(ctx: dict, context: str, model: str) -> str:
    r = grab(ctx["summaries"], context, model)
    return f"{r['mean_seq_rmse']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def main() -> int:
    ctx = run_stage4(ROOT)
    if ctx.get("decision") == "STAGE4_FAIL_LOCK_MISMATCH":
        print("STAGE4_FAIL_LOCK_MISMATCH")
        return 2
    print(f"STAGE4_DECISION={ctx.get('decision')}", flush=True)
    print(f"READY={ctx.get('ready')}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    print(combined, flush=True)
    passed, failed = _parse_pytest(combined)
    _patch_pytest(passed, failed)
    if proc.returncode != 0 and ctx.get("decision") == "PASS":
        print("STAGE4_DECISION=CONDITIONAL_PASS (pytest failures)", flush=True)
    print("NO NEW MODEL WAS TRAINED.")
    print("NO HYPERPARAMETER WAS RETUNED.")
    print("NO TARGET-DOMAIN ADAPTATION WAS PERFORMED.")
    if ctx.get("decision") not in {"PASS", "CONDITIONAL_PASS"}:
        return 1
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
