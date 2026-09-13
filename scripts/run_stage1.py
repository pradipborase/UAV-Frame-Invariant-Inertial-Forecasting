"""Run Stage 1 protocol freeze and causal processing. No modelling."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage1.run import run_stage1  # noqa: E402


def _patch_pytest_counts(passed: str, failed: str) -> None:
    for path in (ROOT / "STAGE1_REPORT.md", ROOT / "results" / "stage1" / "STAGE1_REPORT.md"):
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
    ctx = run_stage1(ROOT)
    print(f"STAGE1_DECISION={ctx['decision']}", flush=True)
    print(f"protocol_sha256={ctx['protocol_sha256']}", flush=True)
    print(f"n_findings={len(ctx['findings'])}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    print(combined, flush=True)
    passed, failed = _parse_pytest(combined)
    _patch_pytest_counts(passed, failed)
    marker = ROOT / ".stage1_complete"
    marker.write_text(
        f"decision={ctx['decision']}\nutc={ctx['utc']}\npytest_passed={passed}\npytest_failed={failed}\n",
        encoding="utf-8",
    )
    stop = ROOT / "STOP_HERE.md"
    block = f"""
# STAGE 1 STATUS

status = {ctx['decision']}
utc = {ctx['utc']}
pytest = {passed} passed, {failed} failed

Protocol lock SHA-256 (PROTOCOL_LOCK.md) = {ctx['protocol_sha256']}

NO MODELLING AUTHORIZED.
NO FORECASTING RESULTS.
STOPPING BEFORE STAGE 2.
"""
    text = stop.read_text(encoding="utf-8") if stop.exists() else ""
    if "# STAGE 1 STATUS" not in text:
        stop.write_text(text.rstrip() + "\n" + block, encoding="utf-8")
    if ctx["decision"] not in {"PASS", "CONDITIONAL_PASS"}:
        return 1
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
