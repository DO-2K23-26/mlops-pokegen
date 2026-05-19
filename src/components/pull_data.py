"""Component for notebook 00_pull_data_dvc.ipynb."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dvc.api import DVCFileSystem

from shared.paths import RAW_CARDS_DIR
from shared.step_artifacts import state_path

from kfp.dsl import component as kfp_component, Output, Artifact



def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


@kfp_component(packages_to_install=["dvc[gs,s3,ssh]"])
def pull_data_component(
    state_output: Output[Artifact],
    repo_url: str = "https://git.razano.dev/llabeyrie/mlops-dataset.git",
    revision: str = "main",
    remote_path: str = "cards",
    local_cards_dir: str = str(RAW_CARDS_DIR),
    dvc_file_path: str = str(RAW_CARDS_DIR.parent / "cards.dvc"),
    aws_default_region: str = "auto",
) -> None:
    """Pull the card dataset from DVC-backed storage and persist the step state."""
    os.environ.setdefault("AWS_DEFAULT_REGION", aws_default_region)

    local_dir = Path(local_cards_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    fs = DVCFileSystem(repo_url, rev=revision)
    fs.get(remote_path, str(local_dir), recursive=True)

    n_png = len(list(fs.glob(f"{remote_path}/**/*.png")))
    payload: dict[str, Any] = {
        "cards_dir": str(local_dir.resolve()),
        "n_png": n_png,
        "dvc_file": str(Path(dvc_file_path).resolve()),
    }

    _write_json(state_output.path, {"step": "00_pull_data", **payload})
    print(f"Pull termine - {n_png} PNG sous {local_dir}")
