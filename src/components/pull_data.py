"""Component for notebook 00_pull_data_dvc.ipynb."""

from kfp.dsl import component as kfp_component, Output, Artifact


@kfp_component(packages_to_install=["dvc[gs,s3,ssh]"])
def pull_data_component(
    state_output: Output[Artifact],
    repo_url: str = "https://git.razano.dev/llabeyrie/mlops-dataset.git",
    revision: str = "main",
    remote_path: str = "cards",
    local_cards_dir: str = "/tmp/data/raw/cards",
    aws_default_region: str = "auto",
) -> None:
    """Pull the card dataset from DVC-backed storage and write the step state.

    AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are injected as env vars
    from a Kubernetes secret (use_secret_as_env in the pipeline definition).
    """
    import json
    import os
    from pathlib import Path

    from dvc.api import DVCFileSystem

    os.environ.setdefault("AWS_DEFAULT_REGION", aws_default_region)

    local_dir = Path(local_cards_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    fs = DVCFileSystem(repo_url, rev=revision)
    fs.get(remote_path, str(local_dir), recursive=True)

    n_png = len(list(fs.glob(f"{remote_path}/**/*.png")))
    payload = {
        "step": "00_pull_data",
        "cards_dir": str(local_dir.resolve()),
        "n_png": n_png,
    }

    out = Path(state_output.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Pull termine - {n_png} PNG sous {local_dir}")
