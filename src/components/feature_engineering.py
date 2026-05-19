"""Component for notebook 01_feature_engineering.ipynb."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CTK_DIR = REPO_ROOT / "clean-text-to-keywords"
if str(CTK_DIR) not in sys.path:
    sys.path.insert(0, str(CTK_DIR))

from keyword_extractor import KeywordExtractor  # type: ignore[import-not-found]
from json_inference import fill_template_from_keywords  # type: ignore[import-not-found]

from shared.step_artifacts import state_path

from kfp.dsl import component as kfp_component, Input, Output, Artifact  # type: ignore[import-not-found]



def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


@kfp_component(packages_to_install=["spacy", "yake"])
def feature_engineering_component(
    input_results_dataset: Input[Artifact],
    metadata_output: Output[Artifact],
    state_output: Output[Artifact],
    user_prompt: str = "A fierce fire dragon pokemon with wings and a blazing tail attack",
    template_path: str = str(CTK_DIR / "json_template_example.json"),
) -> None:
    """Extract keywords from the prompt and build a JSON metadata example."""
    with open(input_results_dataset.path) as f:
        card_images = json.load(f)

    print(f"Dataset (etape 00): {card_images}")

    extractor = KeywordExtractor.from_default_model()
    keywords = extractor.extract(user_prompt)

    template = json.loads(Path(template_path).read_text(encoding="utf-8"))
    metadata = fill_template_from_keywords(template, keywords)

    metadata_path = _write_json(metadata_output.path, metadata)

    payload: dict[str, Any] = {
        "input_results_dataset": str(Path(input_results_dataset.path).resolve()),
        "user_prompt": user_prompt,
        "keywords": keywords,
        "metadata_path": str(metadata_path.resolve()),
    }
    _write_json(state_output.path, {"step": "01_feature_engineering", **payload})
