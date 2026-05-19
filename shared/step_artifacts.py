"""Sauvegarde / chargement des sorties entre étapes de la pipeline MLOps."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlops.shared.paths import PIPELINE_ARTIFACTS_DIR

STEPS_DIR = PIPELINE_ARTIFACTS_DIR / "steps"
STATE_FILENAME = "state.json"

# Ordre canonique des étapes
PIPELINE_STEPS: List[str] = [
    "00_pull_data",
    "01_feature_engineering",
    "02_preprocessing",
    "03_train_model",
    "04_evaluation",
    "05_model_registry",
    "06_deployment",
    "07_monitoring",
]


def step_dir(step_id: str) -> Path:
    if step_id not in PIPELINE_STEPS:
        raise ValueError(f"Étape inconnue: {step_id}. Valides: {PIPELINE_STEPS}")
    return STEPS_DIR / step_id


def state_path(step_id: str) -> Path:
    return step_dir(step_id) / STATE_FILENAME


def previous_step_id(step_id: str) -> Optional[str]:
    idx = PIPELINE_STEPS.index(step_id)
    return PIPELINE_STEPS[idx - 1] if idx > 0 else None


def save_step_output(step_id: str, payload: Dict[str, Any]) -> Path:
    """Enregistre l'état JSON de fin d'étape."""
    out_dir = step_dir(step_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "step": step_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    path = state_path(step_id)
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{step_id}] État sauvegardé → {path}")
    return path


def load_step_output(step_id: str, *, required: bool = True) -> Dict[str, Any]:
    """Charge l'état JSON produit par une étape."""
    path = state_path(step_id)
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Sortie de l'étape '{step_id}' introuvable ({path}). "
                f"Exécutez d'abord le notebook {step_id}."
            )
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"[{step_id}] État chargé ← {path}")
    return data


def load_previous_step_output(step_id: str) -> Dict[str, Any]:
    """Charge la sortie de l'étape immédiatement précédente."""
    prev = previous_step_id(step_id)
    if prev is None:
        raise ValueError(f"L'étape {step_id} n'a pas de prédécesseur.")
    return load_step_output(prev)


def rel_path(path: Path) -> str:
    """Chemin relatif au repo pour sérialisation JSON portable."""
    from mlops.shared.paths import REPO_ROOT

    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_path(rel_or_abs: str) -> Path:
    from mlops.shared.paths import REPO_ROOT

    p = Path(rel_or_abs)
    return p if p.is_absolute() else REPO_ROOT / p
