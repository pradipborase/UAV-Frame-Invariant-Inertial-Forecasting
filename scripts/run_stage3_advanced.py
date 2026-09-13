"""Run Stage 3 compact advanced models. Two-phase freeze then zero-shot transfer."""

from __future__ import annotations

import multiprocessing as mp
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage3.run import run_stage3  # noqa: E402


def _patch_pytest(passed: str, failed: str) -> None:
    for path in (ROOT / "STAGE3_REPORT.md", ROOT / "results" / "stage3" / "STAGE3_REPORT.md"):
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
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    ctx = run_stage3(ROOT)
    if ctx.get("decision") == "PROTOCOL_LOCK_MISMATCH":
        print("PROTOCOL_LOCK_MISMATCH")
        return 2
    print(f"STAGE3_DECISION={ctx.get('decision')}", flush=True)
    print(f"CLASSIFICATION={ctx.get('classification')}", flush=True)
    print(f"freeze_preserved={ctx.get('freeze_preserved')}", flush=True)
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
    if ctx.get("decision") not in {"PASS", "CONDITIONAL_PASS"}:
        return 1
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
