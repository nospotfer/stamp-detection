"""Download candidate datasets from Roboflow Universe in COCO format.

Each URL in datasets.txt is a Roboflow Universe project. We download the latest
version of each into datasets/raw/<workspace>__<project>/ with the standard
Roboflow COCO layout ({train,valid,test}/_annotations.coco.json).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from stamp_detection.utils import require_env


def parse_datasets_file(path: str | Path) -> list[tuple[str, str]]:
    """Parse Roboflow Universe URLs into (workspace, project) pairs.

    Tolerates trailing path segments like /images/<id> or /dataset/<version>.
    """
    pairs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"https?://universe\.roboflow\.com/([^/]+)/([^/]+)", line)
        if not m:
            raise ValueError(f"Cannot parse Roboflow URL: {line}")
        pairs.append((m.group(1), m.group(2)))
    # preserve order, drop accidental duplicates
    seen, unique = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def slug(workspace: str, project: str) -> str:
    return f"{workspace}__{project}"


def download_all(datasets_file: str | Path, raw_dir: str | Path, force: bool = False) -> dict:
    """Download every dataset; returns {slug: version_number}. Idempotent unless force."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=require_env("ROBOFLOW_API_KEY"))
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    versions_path = raw_dir / "versions.json"
    versions = json.loads(versions_path.read_text()) if versions_path.exists() else {}

    for workspace, project_name in parse_datasets_file(datasets_file):
        name = slug(workspace, project_name)
        target = raw_dir / name
        if target.exists() and not force:
            print(f"[download] {name}: already present, skipping")
            continue
        print(f"[download] {name}: fetching latest version...")
        project = rf.workspace(workspace).project(project_name)
        version_list = project.versions()
        if not version_list:
            print(f"[download] {name}: WARNING - no exported versions, skipping")
            continue
        latest = max(version_list, key=lambda v: int(v.version.split("/")[-1]))
        latest.download("coco", location=str(target), overwrite=True)
        versions[name] = int(latest.version.split("/")[-1])
        versions_path.write_text(json.dumps(versions, indent=2))
        print(f"[download] {name}: version {versions[name]} -> {target}")
    return versions
