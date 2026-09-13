"""Run Stage 2 baseline forecasting. No deep learning."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage2.run import run_stage2  # noqa: E402


def _patch_pytest(passed: str, failed: str) -> None:
    for path in (ROOT / "STAGE2_REPORT.md", ROOT / "results" / "stage2" / "STAGE2_REPORT.md"):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace("SEE_PYTEST passed", f"{passed} passed")
        text = text.replace("SEE_PYTEST failed", f"{failed} failed")
        path.write_text(text, encoding="utf-8")


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


def main() -> int:
    ctx = run_stage2(ROOT)
    if ctx.get("decision") == "PROTOCOL_LOCK_MISMATCH":
        print("PROTOCOL_LOCK_MISMATCH")
        return 2
    print(f"STAGE2_DECISION={ctx.get('decision')}", flush=True)
    print(f"ADVANCED_GATE={ctx.get('gate')}", flush=True)
    print(f"transfer_lock={ctx.get('transfer_lock')}", flush=True)
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
    marker = ROOT / ".stage2_complete"
    marker.write_text(
        f"decision={ctx.get('decision')}\ngate={ctx.get('gate')}\npytest_passed={passed}\npytest_failed={failed}\n",
        encoding="utf-8",
    )
    if ctx.get("decision") not in {"PASS", "CONDITIONAL_PASS"}:
        return 1
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
