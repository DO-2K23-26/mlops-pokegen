"""Component for notebook 01_feature_engineering.ipynb."""

from kfp.dsl import component as kfp_component, Input, Output, Artifact

_GIT_PKG = "pokegen-shared @ git+https://github.com/DO-2K23-26/mlops-pokegen.git"

_DEFAULT_TEMPLATE = """{
  "category": "Pokemon", "name": "", "rarity": "", "hp": "", "types": [""],
  "evolveFrom": "", "description": "", "stage": "",
  "attacks": [{"cost": [""], "name": "", "effect": ""},
               {"cost": [""], "name": "", "effect": "", "damage": 0}],
  "weaknesses": [{"type": "", "value": ""}],
  "retreat": 0, "regulationMark": "", "legal": {"standard": true, "expanded": true}
}"""


@kfp_component(packages_to_install=["spacy", "yake", _GIT_PKG])
def feature_engineering_component(
    input_results_dataset: Input[Artifact],
    metadata_output: Output[Artifact],
    state_output: Output[Artifact],
    user_prompt: str = "A fierce fire dragon pokemon with wings and a blazing tail attack",
    template_json: str = _DEFAULT_TEMPLATE,
) -> None:
    """Extract keywords from the prompt and build a JSON metadata profile."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)

    from pokegen_nlp.keyword_extractor import KeywordExtractor
    from pokegen_nlp.json_inference import fill_template_from_keywords

    with open(input_results_dataset.path) as f:
        card_images = json.load(f)
    print(f"Dataset (etape 00): {card_images}")

    extractor = KeywordExtractor.from_default_model()
    keywords = extractor.extract(user_prompt)

    template = json.loads(template_json)
    metadata = fill_template_from_keywords(template, keywords)

    def write_json(path: str, payload: dict) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    metadata_path = write_json(metadata_output.path, metadata)
    write_json(state_output.path, {
        "step": "01_feature_engineering",
        "input_results_dataset": str(Path(input_results_dataset.path).resolve()),
        "user_prompt": user_prompt,
        "keywords": keywords,
        "metadata_path": str(metadata_path.resolve()),
    })
