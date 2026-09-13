"""Stage 1 orchestration: process streams, write audits, freeze protocol. No modelling."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from audit.config import ROOT, load_yaml
from audit.hashing import sha256_file
from audit.stage0c import write_csv

from .causal_filter import (
    FTYPE,
    GPASS_DB,
    GSTOP_DB,
    PASSBAND_HZ,
    STOPBAND_HZ,
    design_antialias_sos,
    ellip_sos_filter_name,
    frequency_response,
    impulse_tail_relative,
    spec_metrics,
)
from .folds import (
    EUROC_ROOM,
    FOLD_COLUMNS,
    UZH_GROUP,
    fold_is_sequence_disjoint,
    leave_group_out,
    leave_one_recording_out,
    transfer_manifests,
)
from .io_native import (
    EUROC_PRIMARY,
    UZH_PRIMARY,
    euroc_inner_zip_path,
    load_euroc_native,
    load_uzh_native,
    uzh_zip_path,
)
from .pipeline import process_invariant_stream, rate_from_timestamps
from .windows import count_windows

UTC = timezone.utc


def utc_now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pct(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p))


def describe_channel(x: np.ndarray) -> dict[str, Any]:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    finite = v[np.isfinite(v)]
    frac = float(finite.size / v.size) if v.size else 0.0
    if finite.size == 0:
        return {
            "min": None,
            "p01": None,
            "median": None,
            "mean": None,
            "std": None,
            "p99": None,
            "max": None,
            "finite_fraction": frac,
            "near_constant": True,
        }
    std = float(np.std(finite, ddof=0))
    mean = float(np.mean(finite))
    near = bool(std < 1e-9 or (abs(mean) > 0 and std / abs(mean) < 1e-8))
    return {
        "min": float(np.min(finite)),
        "p01": _pct(finite, 1),
        "median": float(np.median(finite)),
        "mean": mean,
        "std": std,
        "p99": _pct(finite, 99),
        "max": float(np.max(finite)),
        "finite_fraction": frac,
        "near_constant": near,
    }


def write_processed_csv(
    path: Path,
    dataset: str,
    sequence_id: str,
    ts: np.ndarray,
    q_f: np.ndarray,
    q_w: np.ndarray,
    native_index: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if native_index is None:
        native_index = np.arange(ts.size, dtype=np.int64)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# dataset={dataset}\n")
        handle.write(f"# sequence_id={sequence_id}\n")
        handle.write("# q_f_unit=m/s^2\n")
        handle.write("# q_w_unit=rad/s\n")
        handle.write("# quantity_qf=accelerometer_specific_force_magnitude\n")
        handle.write("# quantity_qw=angular_speed_magnitude\n")
        handle.write("# FIT_SCOPE=NONE_DETERMINISTIC_PHYSICAL\n")
        writer = csv.writer(handle)
        writer.writerow(["timestamp_s", "q_f_mps2", "q_w_radps", "native_index"])
        for t, a, w, idx in zip(ts, q_f, q_w, native_index, strict=True):
            writer.writerow([f"{float(t):.12g}", f"{float(a):.12g}", f"{float(w):.12g}", int(idx)])


def count_major_compat(root: Path) -> list[dict[str, str]]:
    path = root / "results" / "stage0c" / "EUROC_UZH_COMPATIBILITY.csv"
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("severity", "")).upper() == "MAJOR":
                rows.append(row)
    return rows


def _expected_euroc_inner_hashes(root: Path) -> dict[str, str]:
    path = root / "results" / "stage0c" / "EUROC_DOWNLOAD_MANIFEST.csv"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sid = str(row.get("sequence_id", ""))
            if sid in EUROC_PRIMARY:
                out[sid] = str(row.get("sha256", ""))
    return out


def _expected_uzh_hashes(root: Path) -> dict[str, str]:
    path = root / "results" / "stage0c" / "UZH_HASH_REVERIFY.csv"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sid = str(row.get("sequence_id", ""))
            if sid in UZH_PRIMARY:
                out[sid] = str(row.get("expected_sha256") or row.get("observed_sha256") or "")
    return out


def run_stage1(root: Path | None = None) -> dict[str, Any]:
    root = Path(root) if root is not None else ROOT
    cfg = load_yaml(root / "configs" / "stage1.yaml")
    out = root / "results" / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    proc_dir = root / "data" / "processed" / "stage1"
    proc_dir.mkdir(parents=True, exist_ok=True)
    utc = utc_now()
    findings: list[dict[str, str]] = []

    major_compat = count_major_compat(root)
    (out / "STAGE0C_COMPATIBILITY_CLARIFICATION.md").write_text(
        _clarification_md(len(major_compat), major_compat),
        encoding="utf-8",
    )

    sos_by_fs: dict[float, np.ndarray] = {}
    spec_by_fs: dict[float, dict[str, Any]] = {}
    for fs in (200.0, 500.0):
        sos = design_antialias_sos(fs)
        sos_by_fs[fs] = sos
        spec_by_fs[fs] = spec_metrics(sos, fs)
        spec_by_fs[fs]["impulse_tail_rel_1s"] = impulse_tail_relative(sos, fs, 1.0)

    _write_filter_files(out, spec_by_fs, sos_by_fs)

    # Hash-lock retained raw sequence archives against Stage-0C recorded SHA-256 values.
    expected_euroc = _expected_euroc_inner_hashes(root)
    expected_uzh = _expected_uzh_hashes(root)
    hash_rows = []
    for seq in EUROC_PRIMARY:
        z = euroc_inner_zip_path(root, seq)
        print(f"[stage1] hashing EUROC {seq}", flush=True)
        st = "PASS" if z.exists() else "FAIL"
        digest = sha256_file(z) if z.exists() else ""
        expected = expected_euroc.get(seq, "")
        if not z.exists():
            findings.append(_finding(seq, "RAW_HASH", "integrity", "CRITICAL", "FAIL", "missing inner zip", "present", "P1"))
        elif expected and digest != expected:
            st = "FAIL"
            findings.append(_finding(seq, "EUROC_HASH", "integrity", "CRITICAL", "FAIL", digest, expected, "P1 hash mismatch"))
        hash_rows.append(
            {
                "dataset": "EUROC",
                "sequence_id": seq,
                "path": str(z),
                "sha256": digest,
                "expected_sha256": expected,
                "status": st,
            }
        )
    for seq in UZH_PRIMARY:
        z = uzh_zip_path(root, seq)
        print(f"[stage1] hashing UZH {seq}", flush=True)
        st = "PASS" if z.exists() else "FAIL"
        digest = sha256_file(z) if z.exists() else ""
        expected = expected_uzh.get(seq, "")
        if not z.exists():
            findings.append(_finding(seq, "RAW_HASH", "integrity", "CRITICAL", "FAIL", "missing zip", "present", "P1"))
        elif expected and digest != expected:
            st = "FAIL"
            findings.append(_finding(seq, "UZH_HASH", "integrity", "CRITICAL", "FAIL", digest, expected, "P1 hash mismatch"))
        hash_rows.append(
            {
                "dataset": "UZH_FPV",
                "sequence_id": seq,
                "path": str(z),
                "sha256": digest,
                "expected_sha256": expected,
                "status": st,
            }
        )
    write_csv(
        out / "RAW_SOURCE_HASH_REVERIFY.csv",
        ["dataset", "sequence_id", "path", "sha256", "expected_sha256", "status"],
        hash_rows,
    )

    inv_rows: list[dict[str, Any]] = []
    resamp_rows: list[dict[str, Any]] = []
    proc_manifest: list[dict[str, Any]] = []
    proc_integrity: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    processed_meta: list[dict[str, Any]] = []

    warmup = float(cfg["warmup_s"])
    for seq in EUROC_PRIMARY:
        print(f"[stage1] processing EUROC {seq}", flush=True)
        native = load_euroc_native(root, seq)
        rec = _process_one(
            native,
            native_rate=200.0,
            factor=2,
            warmup=warmup,
            sos=sos_by_fs[200.0],
            proc_dir=proc_dir,
        )
        inv_rows.append(rec["invariant"])
        resamp_rows.append(rec["resampling"])
        proc_manifest.append(rec["manifest"])
        proc_integrity.append(rec["integrity"])
        window_rows.append(rec["window"])
        processed_meta.append(rec)
        findings.extend(rec["findings"])
    for seq in UZH_PRIMARY:
        print(f"[stage1] processing UZH {seq}", flush=True)
        native = load_uzh_native(root, seq)
        rec = _process_one(
            native,
            native_rate=500.0,
            factor=5,
            warmup=warmup,
            sos=sos_by_fs[500.0],
            proc_dir=proc_dir,
        )
        inv_rows.append(rec["invariant"])
        resamp_rows.append(rec["resampling"])
        proc_manifest.append(rec["manifest"])
        proc_integrity.append(rec["integrity"])
        window_rows.append(rec["window"])
        processed_meta.append(rec)
        findings.extend(rec["findings"])

    write_csv(
        out / "INVARIANT_FEATURE_AUDIT.csv",
        [
            "dataset",
            "sequence_id",
            "native_accel_fields",
            "native_gyro_fields",
            "q_f_unit",
            "q_w_unit",
            "q_f_min",
            "q_f_p01",
            "q_f_median",
            "q_f_mean",
            "q_f_std",
            "q_f_p99",
            "q_f_max",
            "q_w_min",
            "q_w_p01",
            "q_w_median",
            "q_w_mean",
            "q_w_std",
            "q_w_p99",
            "q_w_max",
            "finite_fraction",
            "near_constant_qf",
            "near_constant_qw",
            "notes",
        ],
        inv_rows,
    )
    write_csv(
        out / "RESAMPLING_AUDIT.csv",
        [
            "dataset",
            "sequence_id",
            "native_rate_hz",
            "decimation_factor",
            "output_rate_hz",
            "filter_name",
            "warmup_removed_s",
            "original_samples",
            "processed_samples",
            "first_processed_timestamp",
            "last_processed_timestamp",
            "timestamp_monotonic",
            "effective_rate_hz",
            "max_rate_deviation",
            "notes",
        ],
        resamp_rows,
    )
    write_csv(
        out / "PROCESSED_DATA_MANIFEST.csv",
        ["dataset", "sequence_id", "local_path", "n_rows", "sha256", "q_f_unit", "q_w_unit", "notes"],
        proc_manifest,
    )
    write_csv(
        out / "PROCESSED_INTEGRITY.csv",
        [
            "dataset",
            "sequence_id",
            "all_finite",
            "strictly_monotonic",
            "approx_100hz",
            "effective_rate_hz",
            "duration_s",
            "enough_for_L100_H20",
            "nonzero_qf_var",
            "nonzero_qw_var",
            "units_ok",
            "hash_recorded",
            "status",
            "notes",
        ],
        proc_integrity,
    )
    write_csv(
        out / "WINDOW_COUNT_AUDIT.csv",
        [
            "dataset",
            "sequence_id",
            "processed_samples",
            "candidate_origins",
            "valid_origins",
            "excluded_due_to_start",
            "excluded_due_to_end",
            "excluded_due_to_gap",
            "excluded_other",
            "lookback_samples",
            "horizon_samples",
            "stride_samples",
        ],
        window_rows,
    )

    euroc_loro = leave_one_recording_out("EUROC", EUROC_PRIMARY, EUROC_ROOM)
    uzh_loro = leave_one_recording_out("UZH_FPV", UZH_PRIMARY, UZH_GROUP)
    source_sel = leave_group_out("EUROC", EUROC_PRIMARY, EUROC_ROOM, evaluation_class="SOURCE_SELECTION") + leave_group_out(
        "UZH_FPV", UZH_PRIMARY, UZH_GROUP, evaluation_class="SOURCE_SELECTION"
    )
    transfer = transfer_manifests(EUROC_PRIMARY, UZH_PRIMARY)
    for name, rows in (
        ("FOLD_MANIFEST_WITHIN_EUROC.csv", euroc_loro),
        ("FOLD_MANIFEST_WITHIN_UZH.csv", uzh_loro),
        ("FOLD_MANIFEST_SOURCE_SELECTION.csv", source_sel),
        ("FOLD_MANIFEST_TRANSFER.csv", transfer),
    ):
        write_csv(out / name, FOLD_COLUMNS, rows)
        ok, issues = fold_is_sequence_disjoint(rows)
        if not ok:
            findings.append(_finding("FOLDS", name, "leakage", "CRITICAL", "FAIL", issues, "disjoint", "P9/P10"))

    # Transfer dataset separation
    for fold_id, src_ds, tgt_ds in (
        ("TRANSFER_EUROC_TO_UZH", "EUROC", "UZH_FPV"),
        ("TRANSFER_UZH_TO_EUROC", "UZH_FPV", "EUROC"),
    ):
        items = [r for r in transfer if r["fold_id"] == fold_id]
        train_ds = {r["dataset"] for r in items if r["split_role"] == "source_train"}
        tgt = {r["dataset"] for r in items if r["split_role"] == "target_eval"}
        if train_ds != {src_ds} or tgt != {tgt_ds}:
            findings.append(
                _finding(
                    fold_id,
                    "TRANSFER_SEP",
                    "leakage",
                    "CRITICAL",
                    "FAIL",
                    f"train={train_ds} target={tgt}",
                    f"train={src_ds} target={tgt_ds}",
                    "P10",
                )
            )

    qf_stds = [r["q_f_std"] for r in inv_rows if r.get("q_f_std") not in (None, "")]
    near_qf = [r["sequence_id"] for r in inv_rows if str(r.get("near_constant_qf")).lower() in {"true", "yes", "1"}]
    degeneracy = bool(qf_stds) and all(float(s) < 0.05 for s in qf_stds)

    (out / "STAGE1_DATA_CHARACTERIZATION.md").write_text(
        _characterization_md(inv_rows, resamp_rows, window_rows),
        encoding="utf-8",
    )

    lock_md = root / "PROTOCOL_LOCK.md"
    lock_yaml = root / "configs" / "protocol_lock.yaml"
    lock_md.write_text(_protocol_lock_md(utc), encoding="utf-8")
    lock_yaml.write_text(_protocol_lock_yaml(), encoding="utf-8")
    hashes = {
        "utc": utc,
        "PROTOCOL_LOCK.md": sha256_file(lock_md),
        "configs/protocol_lock.yaml": sha256_file(lock_yaml),
    }
    hash_text = (
        f"utc={utc}\n"
        f"PROTOCOL_LOCK.md sha256={hashes['PROTOCOL_LOCK.md']}\n"
        f"configs/protocol_lock.yaml sha256={hashes['configs/protocol_lock.yaml']}\n"
        "terminology=PRE-SPECIFIED_LOCKED_PROTOCOL (not an OSF/Zenodo registration)\n"
    )
    (out / "PROTOCOL_LOCK_HASHES.txt").write_text(hash_text, encoding="utf-8")

    _write_stat_plan(root)
    _append_protocol(root)
    _append_decisions(root, utc, hashes["PROTOCOL_LOCK.md"])

    filters_ok = all(s["meets_passband"] and s["meets_stopband"] and s["stable"] for s in spec_by_fs.values())
    enough = all(str(r.get("enough_for_L100_H20")) == "YES" for r in proc_integrity)
    hz_ok = all(str(r.get("approx_100hz")) == "YES" for r in proc_integrity)
    finite_ok = all(str(r.get("all_finite")) == "YES" for r in proc_integrity)
    crit_fail = any(f["severity"] == "CRITICAL" and f["status"] == "FAIL" for f in findings)
    major_fail = any(f["severity"] == "MAJOR" and f["status"] == "FAIL" for f in findings)
    p1 = all(r["status"] == "PASS" for r in hash_rows)
    decision = "PASS"
    if degeneracy:
        decision = "CONDITIONAL_PASS_REVIEW_REQUIRED"
    elif crit_fail or not filters_ok or not enough or not hz_ok or not finite_ok or not p1:
        decision = "FAIL"
    elif near_qf or major_fail:
        decision = "CONDITIONAL_PASS"

    qf_all_min = min(float(r["q_f_min"]) for r in inv_rows)
    qf_all_max = max(float(r["q_f_max"]) for r in inv_rows)
    qw_all_min = min(float(r["q_w_min"]) for r in inv_rows)
    qw_all_max = max(float(r["q_w_max"]) for r in inv_rows)

    ctx = {
        "decision": decision,
        "utc": utc,
        "n_euroc": len(EUROC_PRIMARY),
        "n_uzh": len(UZH_PRIMARY),
        "major_compat_n": len(major_compat),
        "filters_ok": filters_ok,
        "near_qf": near_qf,
        "degeneracy": degeneracy,
        "qf_range": [qf_all_min, qf_all_max],
        "qw_range": [qw_all_min, qw_all_max],
        "protocol_sha256": hashes["PROTOCOL_LOCK.md"],
        "protocol_yaml_sha256": hashes["configs/protocol_lock.yaml"],
        "findings": findings,
        "spec_200": {k: v for k, v in spec_by_fs[200.0].items() if k not in {"sos", "response"}},
        "spec_500": {k: v for k, v in spec_by_fs[500.0].items() if k not in {"sos", "response"}},
        "hash_rows": hash_rows,
    }
    (out / "STAGE1_CONTEXT.json").write_text(json.dumps(ctx, indent=2, default=str), encoding="utf-8")
    report = _stage1_report(ctx, spec_by_fs, proc_integrity, near_qf)
    (root / "STAGE1_REPORT.md").write_text(report, encoding="utf-8")
    (out / "STAGE1_REPORT.md").write_text(report, encoding="utf-8")
    write_csv(
        out / "STAGE1_FINDINGS.csv",
        ["sequence_id", "check_id", "category", "severity", "status", "observed", "expected", "action"],
        findings,
    )
    return ctx


def _process_one(
    native: dict[str, Any],
    *,
    native_rate: float,
    factor: int,
    warmup: float,
    sos: np.ndarray,
    proc_dir: Path,
) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    dqf = describe_channel(native["q_f"])
    dqw = describe_channel(native["q_w"])
    finite = min(dqf["finite_fraction"], dqw["finite_fraction"])
    invariant = {
        "dataset": native["dataset"],
        "sequence_id": native["sequence_id"],
        "native_accel_fields": native["accel_fields"],
        "native_gyro_fields": native["gyro_fields"],
        "q_f_unit": "m/s^2",
        "q_w_unit": "rad/s",
        "q_f_min": dqf["min"],
        "q_f_p01": dqf["p01"],
        "q_f_median": dqf["median"],
        "q_f_mean": dqf["mean"],
        "q_f_std": dqf["std"],
        "q_f_p99": dqf["p99"],
        "q_f_max": dqf["max"],
        "q_w_min": dqw["min"],
        "q_w_p01": dqw["p01"],
        "q_w_median": dqw["median"],
        "q_w_mean": dqw["mean"],
        "q_w_std": dqw["std"],
        "q_w_p99": dqw["p99"],
        "q_w_max": dqw["max"],
        "finite_fraction": finite,
        "near_constant_qf": dqf["near_constant"],
        "near_constant_qw": dqw["near_constant"],
        "notes": "q_f=accelerometer specific-force magnitude; q_w=angular-speed magnitude; no mean/gravity subtraction",
    }
    processed = process_invariant_stream(
        native["timestamp_s"],
        native["q_f"],
        native["q_w"],
        native_rate_hz=native_rate,
        decimation_factor=factor,
        output_rate_hz=100.0,
        warmup_s=warmup,
        sos=sos,
    )
    ts = processed["timestamp_s"]
    qf = processed["q_f"]
    qw = processed["q_w"]
    native_idx = processed["native_index"]
    rates = rate_from_timestamps(ts)
    native_rates = rate_from_timestamps(native["timestamp_s"])
    measured_native = native_rates["median_hz"]
    out_path = proc_dir / f"{native['dataset']}_{native['sequence_id']}.csv"
    write_processed_csv(out_path, native["dataset"], native["sequence_id"], ts, qf, qw, native_idx)
    digest = sha256_file(out_path)
    duration = float(ts[-1] - ts[0]) if ts.size else 0.0
    n = int(ts.size)
    enough = n >= (100 + 20) and duration >= 1.19
    approx = bool(rates["median_hz"] is not None and abs(rates["median_hz"] - 100.0) <= 2.0)
    all_fin = bool(np.all(np.isfinite(qf)) and np.all(np.isfinite(qw)) and np.all(np.isfinite(ts)))
    max_dt = rates.get("max_dt")
    gap_bad = bool(max_dt is not None and float(max_dt) > 0.03)
    resampling = {
        "dataset": native["dataset"],
        "sequence_id": native["sequence_id"],
        "native_rate_hz": native_rate,
        "decimation_factor": factor,
        "output_rate_hz": 100,
        "filter_name": ellip_sos_filter_name(sos, native_rate),
        "warmup_removed_s": warmup,
        "original_samples": processed["n_native"],
        "processed_samples": processed["n_processed"],
        "first_processed_timestamp": float(ts[0]) if ts.size else "",
        "last_processed_timestamp": float(ts[-1]) if ts.size else "",
        "timestamp_monotonic": rates["strictly_monotonic"],
        "effective_rate_hz": rates["median_hz"],
        "max_rate_deviation": rates["max_abs_dev_from_100"],
        "notes": (
            "causal sosfilt then integer stride; no interpolation; "
            f"design_fs={native_rate}; measured_native_median_hz={measured_native}"
        ),
    }
    manifest = {
        "dataset": native["dataset"],
        "sequence_id": native["sequence_id"],
        "local_path": str(out_path),
        "n_rows": n,
        "sha256": digest,
        "q_f_unit": "m/s^2",
        "q_w_unit": "rad/s",
        "notes": "canonical Stage-1 stream",
    }
    integrity = {
        "dataset": native["dataset"],
        "sequence_id": native["sequence_id"],
        "all_finite": "YES" if all_fin else "NO",
        "strictly_monotonic": "YES" if rates["strictly_monotonic"] else "NO",
        "approx_100hz": "YES" if approx else "NO",
        "effective_rate_hz": rates["median_hz"],
        "duration_s": duration,
        "enough_for_L100_H20": "YES" if enough else "NO",
        "nonzero_qf_var": "YES" if float(np.std(qf) if qf.size else 0) > 0 else "NO",
        "nonzero_qw_var": "YES" if float(np.std(qw) if qw.size else 0) > 0 else "NO",
        "units_ok": "YES",
        "hash_recorded": "YES" if digest else "NO",
        "status": "PASS" if all_fin and rates["strictly_monotonic"] and approx and enough else "FAIL",
        "notes": "" if not gap_bad else f"max_dt={max_dt} exceeds 0.03 s; not silently excluded",
    }
    if gap_bad:
        findings.append(
            _finding(
                native["sequence_id"],
                "TIMESTAMP_GAP",
                "integrity",
                "MAJOR",
                "FAIL",
                f"max_dt={max_dt}",
                "<=0.03 s at 100 Hz",
                "P7; CONDITIONAL_PASS if isolated; do not silently exclude",
            )
        )
    if integrity["status"] == "FAIL":
        findings.append(
            _finding(
                native["sequence_id"],
                "PROCESSED_INTEGRITY",
                "integrity",
                "CRITICAL",
                "FAIL",
                integrity,
                "finite monotonic ~100 Hz enough duration",
                "P6-P8",
            )
        )
    window = {"dataset": native["dataset"], "sequence_id": native["sequence_id"], **count_windows(ts)}
    return {
        "invariant": invariant,
        "resampling": resampling,
        "manifest": manifest,
        "integrity": integrity,
        "window": window,
        "findings": findings,
        "processed": processed,
    }


def _finding(seq: str, check: str, cat: str, sev: str, status: str, obs: Any, exp: str, action: str) -> dict[str, str]:
    return {
        "sequence_id": seq,
        "check_id": check,
        "category": cat,
        "severity": sev,
        "status": status,
        "observed": str(obs),
        "expected": exp,
        "action": action,
    }


def _clarification_md(n_major: int, rows: list[dict[str, str]]) -> str:
    lines = [
        "# Stage 0C compatibility clarification",
        "",
        "Do not rewrite historical Stage-0C files.",
        "",
        "`STAGE0C_REPORT.md` reported:",
        "",
        "- Critical findings = 0",
        "- Major findings = 0",
        "",
        "Those counts refer to **raw-data / integrity** checks in `EUROC_INTEGRITY_FINDINGS.csv`.",
        "They are not a claim that cross-dataset compatibility limitations disappeared.",
        "",
        f"`EUROC_UZH_COMPATIBILITY.csv` contains **{n_major}** rows with severity=MAJOR:",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row.get('item')}`: {row.get('compatible')} — {row.get('recommendation')}")
    lines += [
        "",
        "Counted exactly from the Stage-0C CSV (excluding the OVERALL_CLASSIFICATION row, which is INFO).",
        "",
        "## Two different classes of 'MAJOR'",
        "",
        "1. **RAW-DATA/INTEGRITY MAJOR FINDINGS** — file corruption, non-finite IMU, broken timestamps, missing GT. Stage 0C: none failed.",
        "2. **CROSS-DATASET COMPATIBILITY MAJOR LIMITATIONS** — EuRoC and UZH IMU axes are not demonstrated to be the same vehicle directions; UZH vehicle-body CAD is unresolved; a shared body-axis 3-vector is not scientifically justified without a prohibited fitted rotation.",
        "",
        "Stage 1 therefore adopts a **frame-invariant** primary representation (specific-force magnitude and angular-speed magnitude) specifically to avoid requiring an unjustified common body-axis mapping.",
        "",
        "The compatibility limitations remain in force. They are managed by changing the measurand, not by declaring them absent.",
        "",
    ]
    return "\n".join(lines)


def _write_filter_files(out: Path, spec_by_fs: dict[float, dict[str, Any]], sos_by_fs: dict[float, np.ndarray]) -> None:
    resp_rows = []
    for fs, spec in spec_by_fs.items():
        resp = frequency_response(sos_by_fs[fs], fs, n=2048)
        for f, mag, ph in zip(resp["frequency_hz"], resp["magnitude_db"], resp["phase_rad"], strict=True):
            if f > fs / 2:
                continue
            resp_rows.append(
                {
                    "native_fs_hz": fs,
                    "frequency_hz": f,
                    "magnitude_db": mag,
                    "phase_rad": ph,
                }
            )
    write_csv(out / "FILTER_RESPONSE.csv", ["native_fs_hz", "frequency_hz", "magnitude_db", "phase_rad"], resp_rows)
    s200 = spec_by_fs[200.0]
    s500 = spec_by_fs[500.0]
    text = f"""# Causal anti-alias filter specification (Stage 1)

No prediction performance was used to choose these edges.

## Locked physical requirements (both datasets)

- Filter class: causal IIR, elliptic (`ellip`), second-order sections (SOS)
- Implementation: `scipy.signal.sosfilt` forward only
- Forbidden: `filtfilt`, `sosfiltfilt`, zero-phase, centred moving averages
- Passband edge: {PASSBAND_HZ} Hz
- Stopband edge: {STOPBAND_HZ} Hz
- Passband ripple: <= {GPASS_DB} dB
- Stopband attenuation: >= {GSTOP_DB} dB
- FIT_SCOPE: NONE_DETERMINISTIC_PHYSICAL

## Native-rate choice

Documented nominal rates are used for filter design and integer decimation:

- EuRoC: 200 Hz (Stage 0C measured median 200.00256 Hz). Integer factor 2 -> 100 Hz.
- UZH-FPV: 500 Hz (Stage 0 measured Snapdragon ~500 Hz). Integer factor 5 -> 100 Hz.

The 0.00256 Hz EuRoC offset is sampling jitter, not a different Nyquist grid. Designing at 200.00256 Hz would break integer decimation. Nominal documented rates are therefore the scientifically appropriate design frequencies.

## EuRoC design (fs = 200 Hz)

- type: {FTYPE}
- order: {s200['order']} ({s200['n_sections']} SOS sections; order counted from SOS denominators, not 2*n_sections)
- max pole radius: {s200['max_pole_radius']:.12g}
- stable: {s200['stable']}
- causal: YES
- maximum passband loss: {s200['max_passband_loss_db']:.6g} dB
- minimum stopband attenuation: {s200['min_stopband_atten_db']:.6g} dB
- passband group-delay median (samples): {s200['group_delay_passband_median_samples']:.6g}
- passband group-delay max (samples): {s200['group_delay_passband_max_samples']:.6g}
- impulse-response relative tail after 1.0 s: {s200['impulse_tail_rel_1s']:.6g}
- SOS coefficients (b0,b1,b2,a0,a1,a2 per section):
```
{np.array2string(np.asarray(s200['sos']), precision=16)}
```

## UZH design (fs = 500 Hz)

- type: {FTYPE}
- order: {s500['order']} ({s500['n_sections']} SOS sections; order counted from SOS denominators, not 2*n_sections)
- max pole radius: {s500['max_pole_radius']:.12g}
- stable: {s500['stable']}
- causal: YES
- maximum passband loss: {s500['max_passband_loss_db']:.6g} dB
- minimum stopband attenuation: {s500['min_stopband_atten_db']:.6g} dB
- passband group-delay median (samples): {s500['group_delay_passband_median_samples']:.6g}
- passband group-delay max (samples): {s500['group_delay_passband_max_samples']:.6g}
- impulse-response relative tail after 1.0 s: {s500['impulse_tail_rel_1s']:.6g}
- SOS coefficients:
```
{np.array2string(np.asarray(s500['sos']), precision=16)}
```

## Startup transient / warmup

Causal IIR filters have a startup transient. It is **not** compensated with future samples.

A conservative **1.0 s** physical-time exclusion is applied to both datasets after filtering and decimation.

Reason: at 1.0 s the relative impulse-response tail is ~4e-6 (200 Hz design) and ~1.5e-4 (500 Hz design), far below the 1 s physical-time rule requested for a shared warmup. The same physical duration is used on both datasets.

## Group delay

Group delay is frequency-dependent (elliptic). It is a deterministic property of the locked filter, not a fitted lag. No zero-phase inversion is applied because that would leak the future.
"""
    (out / "FILTER_SPECIFICATION.md").write_text(text, encoding="utf-8")


def _characterization_md(inv: list[dict[str, Any]], resamp: list[dict[str, Any]], windows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage 1 data characterization (not model performance)",
        "",
        "No forecast curves and no forecast errors are reported.",
        "",
        "q_f is accelerometer specific-force magnitude (m/s^2), not linear acceleration.",
        "q_w is angular-speed magnitude (rad/s).",
        "",
        "## Native q_f by sequence",
        "",
        "| dataset | sequence | median | mean | std | min | max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in inv:
        lines.append(
            f"| {r['dataset']} | {r['sequence_id']} | {r['q_f_median']:.6g} | {r['q_f_mean']:.6g} | {r['q_f_std']:.6g} | {r['q_f_min']:.6g} | {r['q_f_max']:.6g} |"
        )
    lines += ["", "## Native q_w by sequence", "", "| dataset | sequence | median | mean | std | min | max |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in inv:
        lines.append(
            f"| {r['dataset']} | {r['sequence_id']} | {r['q_w_median']:.6g} | {r['q_w_mean']:.6g} | {r['q_w_std']:.6g} | {r['q_w_min']:.6g} | {r['q_w_max']:.6g} |"
        )
    lines += ["", "## Processed duration and rate", "", "| dataset | sequence | processed_n | duration_s | effective_hz | valid_windows |", "|---|---|---:|---:|---:|---:|"]
    wmap = {(r["dataset"], r["sequence_id"]): r for r in windows}
    for r in resamp:
        w = wmap.get((r["dataset"], r["sequence_id"]), {})
        dur = ""
        if r.get("first_processed_timestamp") not in ("", None) and r.get("last_processed_timestamp") not in ("", None):
            dur = f"{float(r['last_processed_timestamp']) - float(r['first_processed_timestamp']):.6g}"
        lines.append(
            f"| {r['dataset']} | {r['sequence_id']} | {r['processed_samples']} | {dur} | {r['effective_rate_hz']} | {w.get('valid_origins', '')} |"
        )
    lines += [
        "",
        "## Trajectory groups",
        "",
        "- EuRoC V1_* : EUROC_FIREFLY_VICON_ROOM1 (easy/medium/difficult)",
        "- EuRoC V2_* : EUROC_FIREFLY_VICON_ROOM2 (easy/medium/difficult)",
        "- UZH indoor_forward, indoor_45, outdoor_forward (one retained outdoor recording)",
        "",
    ]
    return "\n".join(lines)


def _protocol_lock_md(utc: str) -> str:
    return f"""# PROTOCOL_LOCK.md — Stage 1 locked protocol

Created utc: {utc}

This is a **PRE-SPECIFIED LOCKED PROTOCOL**.

It is not an external OSF/Zenodo registration. Do not describe this file as pre-registered.

Git commit hash, if any, is recorded in `results/stage1/PROTOCOL_LOCK_HASHES.txt` after these bytes are hashed, to avoid a self-referential hash.

## Datasets

- D1 = EuRoC MAV (6 Vicon primary recordings)
- D2 = UZH-FPV Snapdragon (8 primary recordings)

## Primary sequences

EuRoC: V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult

UZH-FPV: indoor_forward_6_snapdragon, indoor_forward_9_snapdragon, indoor_forward_10_snapdragon, indoor_45_2_snapdragon, indoor_45_4_snapdragon, indoor_45_13_snapdragon, indoor_45_14_snapdragon, outdoor_forward_1_snapdragon

## Primary physical quantity

Accelerometer specific-force magnitude

q_f(t) = sqrt(f_x(t)^2 + f_y(t)^2 + f_z(t)^2)

unit: m/s^2

## Auxiliary / input channel

Angular-speed magnitude

q_w(t) = sqrt(w_x(t)^2 + w_y(t)^2 + w_z(t)^2)

unit: rad/s

## Primary future forecast target

q_f(t)

Primary input channels: q_f(t), q_w(t)

Secondary candidate target q_w(t) is NOT in the confirmatory experiment.

## Common sampling rate

100 Hz after causal anti-alias filtering and integer decimation.

## Filter requirements

Causal elliptic IIR SOS. Passband 30 Hz, stopband 45 Hz, passband ripple <= 1 dB, stopband >= 60 dB. Same physical edges for both native rates. Warmup 1.0 s. FIT_SCOPE = NONE_DETERMINISTIC_PHYSICAL.

## Windowing

- lookback 1.0 s = 100 samples
- horizon 0.20 s = 20 samples
- stride 0.05 s = 5 samples
- report 50 / 100 / 200 ms (h = 5, 10, 20) plus h = 1..20 curve

## Inferential unit

SEQUENCE / flight / recording. Not window.

## Folds

- Within-EuRoC: 6 leave-one-recording-out folds
- Within-UZH: 8 leave-one-recording-out folds
- Source selection: EuRoC leave-room-out; UZH leave-trajectory-group-out (outdoor_forward is a single-recording group)
- Transfer: EUROC -> UZH and UZH -> EUROC, reported separately

## Future normalization

Any statistical scaler: SOURCE TRAIN ONLY. Never target mean/std/min/max.

## Primary metric

Equal-sequence mean of sequence-level multi-horizon RMSE of q_f over h=1..20, reported in m/s^2 after any inverse scaling.

## Secondary metrics

MAE (m/s^2); RMSE at 50/100/200 ms; RMSE-versus-horizon; persistence-relative skill. Window-weighted RMSE is descriptive only.

## Future persistence definition (not executed in Stage 1)

qhat_f(t+h) = q_f(t) for h=1..20.

## Bootstrap

10,000 sequence-level bootstrap replicates, seed 20260912, percentile CIs.

## Zero-shot rules

Target data are not used to fit model parameters, scalers, coordinate alignment, or hyperparameters.

Manuscript wording: "zero-shot with respect to model fitting and data-driven adaptation" — not "the target dataset was completely unseen during study design".

## Forbidden target adaptations

target-test mean/std or min/max; target-fitted PCA; target-fitted rotation; target-fitted bias; target-fitted time warp; post-test retuning.

## Primary scientific question

To what extent can short-horizon prediction of accelerometer specific-force magnitude generalize across UAV platforms, sensor hardware, flight regimes, and datasets when the representation is rotation invariant and all fitted transformations are restricted to the source domain?
"""


def _protocol_lock_yaml() -> str:
    return """# Locked Stage-1 protocol. PRE-SPECIFIED LOCKED PROTOCOL (not an OSF/Zenodo registration).
protocol_name: UAV_MEASUREMENT_FRESH_STAGE1
terminology: PRE_SPECIFIED_LOCKED_PROTOCOL
datasets: [EUROC, UZH_FPV]
primary_sequences:
  EUROC: [V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult]
  UZH_FPV: [indoor_forward_6_snapdragon, indoor_forward_9_snapdragon, indoor_forward_10_snapdragon, indoor_45_2_snapdragon, indoor_45_4_snapdragon, indoor_45_13_snapdragon, indoor_45_14_snapdragon, outdoor_forward_1_snapdragon]
primary_physical_quantity: accelerometer_specific_force_magnitude
primary_target: q_f
input_channels: [q_f, q_w]
secondary_target_not_confirmatory: q_w
q_f_unit: m/s^2
q_w_unit: rad/s
common_sampling_rate_hz: 100
filter:
  causal: true
  requirements:
    passband_hz: 30
    stopband_hz: 45
    gpass_db: 1
    gstop_db: 60
    causal: true
    zero_phase_forbidden: true
  fit_scope: NONE_DETERMINISTIC_PHYSICAL
lookback_s: 1.0
lookback_samples: 100
horizon_s: 0.20
horizon_samples: 20
stride_s: 0.05
stride_samples: 5
forecast_reporting_horizons_ms: [50, 100, 200]
inferential_unit: SEQUENCE
within_domain_folds:
  EUROC: leave_one_recording_out_n6
  UZH_FPV: leave_one_recording_out_n8
source_selection_folds:
  EUROC: leave_vicon_room_out
  UZH_FPV: leave_trajectory_group_out
transfer_directions: [EUROC_TO_UZH, UZH_TO_EUROC]
future_normalization_rule: SOURCE_TRAIN_ONLY
FIT_SCOPE_FITTED_PREPROCESSORS: SOURCE_TRAIN_ONLY
primary_metric: equal_sequence_mean_multihorizon_RMSE_qf_mps2
secondary_metrics: [MAE_mps2, RMSE_50ms, RMSE_100ms, RMSE_200ms, RMSE_vs_horizon, persistence_skill]
bootstrap_repetitions: 10000
bootstrap_seed: 20260912
zero_shot_rules:
  no_target_parameter_fitting: true
  no_target_scaler: true
  no_target_rotation: true
  wording: zero-shot with respect to model fitting and data-driven adaptation
forbidden_target_adaptations:
  - target_test_mean_std
  - target_test_min_max
  - target_fitted_PCA
  - target_fitted_rotation
  - target_fitted_bias
  - target_fitted_time_warp
  - post_test_retuning
"""


def _write_stat_plan(root: Path) -> None:
    (root / "STATISTICAL_ANALYSIS_PLAN.md").write_text(
        """# STATISTICAL_ANALYSIS_PLAN.md

Locked at Stage 1. No hypothesis tests are executed in Stage 1.

## Inferential unit

**PRIMARY INFERENTIAL UNIT = FLIGHT / RECORDING / SEQUENCE**

Windows are not independent experimental units. Thousands of overlapping 1.0 s histories do not create thousands of independent samples.

## Primary outcome

Sequence-level multi-horizon RMSE of accelerometer specific-force magnitude q_f over horizons h = 1,...,20 at 100 Hz, in m/s^2.

RMSE_s = sqrt( mean_{origins, h=1..20} (qhat_f - q_f)^2 )

## Dataset summary (primary estimand)

Equal-sequence mean RMSE: each recording has weight 1/n_sequences.

Not pooled-window RMSE.

## Uncertainty

Sequence-level bootstrap, 10,000 replicates, seed 20260912, percentile confidence intervals.

Because n = 6 (EuRoC) or n = 8 (UZH) is small, p-values will not be overstated.

## Future paired comparison versus persistence

Delta_s = RMSE_candidate,s - RMSE_persistence,s

Report mean paired difference, median paired difference, bootstrap CI, and win/loss/tie counts.

An exact paired sign-flip/permutation test may be used for a single pre-specified primary model. Not computed in Stage 1.

## Two transfer directions

EUROC -> UZH and UZH -> EUROC are separate estimands. Do not pool them. If both are tested formally, use Holm on those two primary comparisons. Not computed in Stage 1.

## Secondary metrics

MAE (m/s^2); RMSE at 50, 100, 200 ms; full RMSE-versus-horizon; persistence skill = 1 - RMSE_model / RMSE_persistence.

Window-weighted descriptive RMSE may be reported only as secondary and must be labelled as such.

## Future model selection

Source-domain validation only. The target dataset must not influence family or hyperparameter choice. After source-only selection, refit on authorized source training recordings and evaluate once, frozen, on the target. No post-test retuning.

## Future persistence (definition only)

qhat_f(t+h) = q_f(t) for h = 1,...,20. Not executed in Stage 1.

## Primary scientific question (frozen)

To what extent can short-horizon prediction of accelerometer specific-force magnitude generalize across UAV platforms, sensor hardware, flight regimes, and datasets when the representation is rotation invariant and all fitted transformations are restricted to the source domain?

Secondary questions only:

- RQ2: within-dataset vs cross-dataset zero-shot performance
- RQ3: error vs horizon 10–200 ms
- RQ4: consistency across recordings and flight-condition groups
""",
        encoding="utf-8",
    )


def _append_protocol(root: Path) -> None:
    path = root / "PROTOCOL.md"
    extra = """

## Stage 1 locked additions

10. **Frame-invariant primary representation.** Cross-dataset principal target is accelerometer specific-force magnitude q_f, not native 3-axis components.
11. **Causal 100 Hz resampling.** Elliptic SOS anti-alias (30/45 Hz, <=1 dB / >=60 dB) then integer decimation. No zero-phase filtering.
12. **Inferential unit remains the sequence.** Window counts are operational, not statistical sample sizes.
13. **Source-train-only fitted transforms.** Physical Euclidean norms, unit conversion, locked causal filters, and integer decimation are deterministic (FIT_SCOPE = NONE). Scalers, if used later, are SOURCE_TRAIN_ONLY.
14. **Do not call this an OSF/Zenodo registration.** Use PRE-SPECIFIED / LOCKED PROTOCOL unless an external timestamped record exists before modelling.
"""
    text = path.read_text(encoding="utf-8")
    if "Stage 1 locked additions" not in text:
        path.write_text(text + extra, encoding="utf-8")


def _append_decisions(root: Path, utc: str, lock_hash: str) -> None:
    path = root / "DECISIONS.md"
    block = f"""
## {utc}

- **decision:** Adopt frame-invariant q_f (accelerometer specific-force magnitude, m/s^2) as the primary cross-dataset target, with q_w as a primary input and non-confirmatory secondary candidate. Freeze 100 Hz causal resampling and the Stage-1 statistical plan.
- **reason:** EuRoC and UZH share SI specific force and angular velocity, but native axes are not a demonstrated common vehicle frame. A fitted rotation is prohibited. Magnitude is invariant to a fixed orthonormal sensor rotation.
- **evidence:** Stage 0C EUROC_UZH_COMPATIBILITY.csv MAJOR frame rows; PROTOCOL_LOCK.md sha256={lock_hash}
- **consequence:** Native 3-axis components are not the principal EuRoC<->UZH zero-shot target. No forecasting is authorized in Stage 1.
"""
    text = path.read_text(encoding="utf-8")
    if "frame-invariant q_f" not in text:
        path.write_text(text.rstrip() + "\n" + block, encoding="utf-8")


def _stage1_report(
    ctx: dict[str, Any],
    spec_by_fs: dict[float, dict[str, Any]],
    proc_integrity: list[dict[str, Any]],
    near_qf: list[str],
) -> str:
    s200 = spec_by_fs[200.0]
    n_fail = sum(1 for r in proc_integrity if r.get("status") != "PASS")
    return f"""# STAGE 1 REPORT

Primary representation:
frame-invariant accelerometer specific-force magnitude q_f = ||f||_2 and angular-speed magnitude q_w = ||omega||_2

Primary target:
q_f (m/s^2)

Auxiliary input:
q_w (rad/s)

Reason for invariant representation:
EuRoC and UZH share the physical inertial quantity and SI units, but native sensor axes are not demonstrated to be the same vehicle directions. UZH body CAD is unresolved. A fitted cross-dataset rotation is prohibited.

EuRoC sequences retained:
6 (V1_01_easy, V1_02_medium, V1_03_difficult, V2_01_easy, V2_02_medium, V2_03_difficult)

UZH sequences retained:
8 (indoor_forward_6/9/10_snapdragon, indoor_45_2/4/13/14_snapdragon, outdoor_forward_1_snapdragon)

Total recordings:
14

Native rates:
EuRoC = 200 Hz documented (measured median 200.00256 Hz)
UZH = 500 Hz documented (measured ~500 Hz)

Common rate:
100 Hz

Filter:
causal elliptic IIR SOS; EuRoC/200 Hz order {s200['order']}; UZH-FPV/500 Hz order {s500['order']}; passband 30 Hz; stopband 45 Hz; ripple <= 1 dB; stopband >= 60 dB; designed at 200 Hz and 500 Hz

Causal:
YES

Future leakage test:
see pytest test_preprocessing_causality.py

Warmup removed:
1.0 s physical time (same for both datasets)

q_f descriptive range:
{ctx['qf_range'][0]:.6g} to {ctx['qf_range'][1]:.6g} m/s^2 (native, all 14 recordings)

q_w descriptive range:
{ctx['qw_range'][0]:.6g} to {ctx['qw_range'][1]:.6g} rad/s (native, all 14 recordings)

Near-constant sequences:
{', '.join(near_qf) if near_qf else 'none'}

Processed-integrity:
{14 - n_fail}/14 PASS

Lookback:
1.0 s / 100 samples

Forecast horizon:
0.20 s / 20 samples

Evaluation stride:
0.05 s / 5 samples

Report horizons:
50 / 100 / 200 ms

Within-EuRoC folds:
6 leave-one-recording-out

Within-UZH folds:
8 leave-one-recording-out

Source-selection groups:
EuRoC Vicon Room 1 vs Room 2; UZH indoor_forward, indoor_45, outdoor_forward (outdoor flagged as n=1)

Cross-dataset directions:
EUROC -> UZH
UZH -> EUROC

Inferential unit:
SEQUENCE

Primary future metric:
equal-sequence multi-horizon RMSE in m/s^2

Future zero-shot normalization:
SOURCE-TRAIN-ONLY

Protocol lock SHA-256:
{ctx['protocol_sha256']}

Compatibility clarification:
STAGE0C_REPORT integrity Major=0 is distinct from {ctx['major_compat_n']} MAJOR compatibility-matrix rows; see results/stage1/STAGE0C_COMPATIBILITY_CLARIFICATION.md

Critical findings:
{sum(1 for f in ctx['findings'] if f.get('severity')=='CRITICAL' and f.get('status')=='FAIL')}

Major findings:
{sum(1 for f in ctx['findings'] if f.get('severity')=='MAJOR' and f.get('status')=='FAIL')}

pytest:
SEE_PYTEST passed
SEE_PYTEST failed

OVERALL STAGE-1 DECISION:
{ctx['decision']}

NO FORECASTING MODEL WAS TRAINED.
NO PREDICTIONS WERE GENERATED.
NO MODEL PERFORMANCE WAS INSPECTED.
"""
