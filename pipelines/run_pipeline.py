#!/usr/bin/env python3
"""
Orchestration du flux MLOps : exécute les notebooks .ipynb de chaque étape (nbconvert).

Usage:
  python mlops/pipelines/run_pipeline.py
  python mlops/pipelines/run_pipeline.py --from-step 03_train_model --dry-run
  python mlops/pipelines/run_pipeline.py --execute  # nécessite jupyter + dépendances GPU
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MLOPS_DIR = REPO_ROOT / "mlops"

STEPS: list[tuple[str, Path]] = [
    ("00_pull_data", MLOPS_DIR / "00_pull_data" / "00_pull_data_dvc.ipynb"),
    ("01_feature_engineering", MLOPS_DIR / "01_feature_engineering" / "01_feature_engineering.ipynb"),
    ("02_preprocessing", MLOPS_DIR / "02_preprocessing" / "02_preprocessing.ipynb"),
    ("03_train_model", MLOPS_DIR / "03_train_model" / "03_train_model.ipynb"),
    ("04_evaluation", MLOPS_DIR / "04_evaluation" / "04_evaluation.ipynb"),
    ("05_model_registry", MLOPS_DIR / "05_model_registry" / "05_model_registry.ipynb"),
    ("06_deployment", MLOPS_DIR / "06_deployment" / "06_deployment.ipynb"),
    ("07_monitoring", MLOPS_DIR / "07_monitoring" / "07_monitoring.ipynb"),
]


def run_notebook(nb_path: Path, cwd: Path, timeout: int | None) -> int:
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={timeout or 0}",
        "--inplace",
        str(nb_path),
    ]
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline MLOps (notebooks séquentiels).")
    parser.add_argument(
        "--from-step",
        default=None,
        help="Étape de départ (ex: 03_train_model).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Exécuter réellement les notebooks (sinon liste seulement).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout nbconvert par notebook (secondes, 0 = illimité).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias de l'exécution sans --execute (liste les notebooks).",
    )
    args = parser.parse_args()

    start_idx = 0
    if args.from_step:
        names = [n for n, _ in STEPS]
        if args.from_step not in names:
            raise SystemExit(f"Étape inconnue: {args.from_step}. Valides: {names}")
        start_idx = names.index(args.from_step)

    do_execute = args.execute and not args.dry_run

    for name, nb_path in STEPS[start_idx:]:
        print(f"\n>>> {name}")
        if not nb_path.exists():
            raise SystemExit(f"Notebook introuvable: {nb_path}")
        if not do_execute:
            print(f"    {nb_path.relative_to(REPO_ROOT)}")
            continue
        code = run_notebook(nb_path, cwd=REPO_ROOT, timeout=args.timeout)
        if code != 0:
            raise SystemExit(f"Échec {name} (code {code})")

    if not do_execute:
        print("\nMode liste. Relancer avec --execute pour exécuter les notebooks.")


if __name__ == "__main__":
    main()
