"""Live verification of official sources. No unofficial mirrors."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .config import Stage0Config
from .downloader import utc_now

UTC = timezone.utc


def probe_url(url: str, timeout_s: int = 25) -> dict[str, Any]:
    result: dict[str, Any] = {
        "url": url,
        "ok": False,
        "status_code": None,
        "error": "",
        "content_length": None,
        "server": "",
        "classification": "UNRESOLVED",
    }
    try:
        response = requests.head(url, timeout=timeout_s, allow_redirects=True)
        result["status_code"] = int(response.status_code)
        result["ok"] = 200 <= response.status_code < 400
        length = response.headers.get("Content-Length")
        result["content_length"] = int(length) if length and str(length).isdigit() else None
        result["server"] = response.headers.get("Server") or ""
        if result["ok"]:
            result["classification"] = "OFFICIAL_SOURCE_REACHABLE"
        elif response.status_code in (502, 503, 504):
            result["classification"] = "OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE"
        elif response.status_code == 404:
            result["classification"] = "OFFICIAL_URL_NOT_FOUND"
        else:
            result["classification"] = f"HTTP_{response.status_code}"
    except requests.exceptions.ConnectionError as exc:
        text = str(exc)
        result["error"] = text
        if "Failed to resolve" in text or "Name or service not known" in text or "getaddrinfo" in text.lower() or "not be resolved" in text.lower():
            result["classification"] = "OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE"
        else:
            result["classification"] = "OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE"
    except requests.RequestException as exc:
        result["error"] = str(exc)
        result["classification"] = "OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE"
    return result


def host_is_official(url: str) -> bool:
    host = urlparse(url).hostname or ""
    allowed = (
        "blackbird-dataset.mit.edu",
        "aera.mit.edu",
        "github.com",
        "fpv.ifi.uzh.ch",
        "rpg.ifi.uzh.ch",
        "download.ifi.uzh.ch",
        "doi.org",
        "arxiv.org",
        "ar5iv.labs.arxiv.org",
    )
    return any(host == item or host.endswith("." + item) for item in allowed)


def verify_sources(cfg: Stage0Config) -> dict[str, Any]:
    sources = cfg.sources
    report: dict[str, Any] = {
        "access_utc": utc_now(),
        "blackbird": {"probes": [], "official_source_verified": False, "data_host_status": "UNRESOLVED"},
        "uzh_fpv": {"probes": [], "official_source_verified": False, "data_host_status": "UNRESOLVED"},
    }
    for key in ("blackbird", "uzh_fpv"):
        urls = sources.get("probe_urls", {}).get(key, [])
        probes = [probe_url(url) for url in urls]
        report[key]["probes"] = probes
        reachable = [p for p in probes if p.get("ok")]
        report[key]["official_source_verified"] = bool(reachable)

    bb_data = [p for p in report["blackbird"]["probes"] if "blackbird-dataset.mit.edu" in p["url"]]
    if bb_data and all(p["classification"] == "OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE" for p in bb_data):
        report["blackbird"]["data_host_status"] = "OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE"
    elif any(p.get("ok") and "BlackbirdDatasetData" in p["url"] for p in report["blackbird"]["probes"]):
        report["blackbird"]["data_host_status"] = "OFFICIAL_SOURCE_REACHABLE"
    else:
        report["blackbird"]["data_host_status"] = "OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE"

    uzh_data = [p for p in report["uzh_fpv"]["probes"] if "rpg.ifi.uzh.ch/datasets" in p["url"] or "fpv.ifi.uzh.ch" in p["url"]]
    if any(p.get("ok") for p in uzh_data):
        report["uzh_fpv"]["data_host_status"] = "OFFICIAL_SOURCE_REACHABLE"
    else:
        report["uzh_fpv"]["data_host_status"] = "OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE"

    gh = cfg.blackbird_tools
    report["blackbird"]["tools_clone"] = {
        "path": str(gh),
        "present": gh.exists(),
        "origin": "https://github.com/mit-aera/Blackbird-Dataset",
    }
    git_head = gh / ".git" / "HEAD"
    if git_head.exists():
        report["blackbird"]["tools_clone"]["head_ref"] = git_head.read_text(encoding="utf-8").strip()
    return report


def render_source_verification_md(report: dict[str, Any], cfg: Stage0Config) -> str:
    bb = cfg.sources["blackbird"]
    uzh = cfg.sources["uzh_fpv"]
    lines = [
        "# Stage 0 — Source verification",
        "",
        f"Access UTC: {report.get('access_utc', '')}",
        "",
        "Unofficial mirrors were not used.",
        "",
        "## MIT Blackbird UAV Dataset",
        "",
        f"- Official institution: {bb['official_institution']}",
        f"- Official website: {bb['official_website_dataset']}",
        f"- Official repository: {bb['official_repository']}",
        f"- Data host status: `{report['blackbird']['data_host_status']}`",
        f"- Metadata/tools GitHub reachable: {report['blackbird']['official_source_verified']}",
        "",
        "### Probes",
        "",
    ]
    for probe in report["blackbird"]["probes"]:
        lines.append(
            f"- `{probe['url']}` status={probe.get('status_code')} class=`{probe.get('classification')}` error={probe.get('error') or 'none'}"
        )
    lines.extend(
        [
            "",
            "## UZH-FPV Drone Racing Dataset",
            "",
            f"- Official institution: {uzh['official_institution']}",
            f"- Official website: {uzh['official_website']}",
            f"- Dataset files: {uzh['official_dataset_files']}",
            f"- Data host status: `{report['uzh_fpv']['data_host_status']}`",
            f"- Official pages/files reachable: {report['uzh_fpv']['official_source_verified']}",
            "",
            "### Probes",
            "",
        ]
    )
    for probe in report["uzh_fpv"]["probes"]:
        lines.append(
            f"- `{probe['url']}` status={probe.get('status_code')} class=`{probe.get('classification')}` error={probe.get('error') or 'none'}"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "If Blackbird's official data host is unavailable, Stage 0 records",
            "`OFFICIAL_SOURCE_TEMPORARILY_UNAVAILABLE` and does not substitute Kaggle,",
            "AcademicTorrents, or other unofficial copies.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def write_source_verification(cfg: Stage0Config, report: dict[str, Any]) -> Path:
    path = cfg.results_dir / "SOURCE_VERIFICATION.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_source_verification_md(report, cfg), encoding="utf-8")
    return path
