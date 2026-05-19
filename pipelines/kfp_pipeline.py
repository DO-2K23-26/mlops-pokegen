"""
Kubeflow Pipelines v2 definition for the Pokémon card LoRA pipeline.

Compile to YAML:
    python pipelines/kfp_pipeline.py

Upload and run on a live cluster:
    python pipelines/kfp_pipeline.py --host https://<kfp-host> [--run]
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kfp import compiler, dsl, kubernetes
from kfp.client import Client

from src.components import (
    evaluation_component,
    feature_engineering_component,
    preprocessing_component,
    pull_data_component,
    train_model_component,
)

PIPELINE_YAML = REPO_ROOT / "pipelines" / "pokegen_pipeline.yaml"


@dsl.pipeline(
    name="pokegen-mlops",
    description="SD 1.5 LoRA fine-tune on Pokémon cards — pull → FE → preprocess → train → eval",
)
def pokegen_pipeline(
    # --- data ---
    repo_url: str = "https://git.razano.dev/llabeyrie/mlops-dataset.git",
    revision: str = "main",
    r2_secret_name: str = "cloudflare-r2-keys",
    # --- feature engineering ---
    user_prompt: str = "A fierce fire dragon pokemon with wings and a blazing tail attack",
    # --- preprocessing ---
    img_size: int = 512,
    batch_size: int = 4,
    val_fraction: float = 0.1,
    split_seed: int = 42,
    # --- training ---
    model_id: str = "runwayml/stable-diffusion-v1-5",
    epochs: int = 1,
    lr: float = 1e-5,
    train_fraction: float = 1.0,
    # --- evaluation ---
    num_inference_steps: int = 10,
    guidance_scale: float = 7.5,
    sample_count: int = 4,
) -> None:
    pull = pull_data_component(
        repo_url=repo_url,
        revision=revision,
    )

    kubernetes.use_secret_as_env(
        pull,
        secret_name=r2_secret_name,
        secret_key_to_env={"access_key_id": "AWS_ACCESS_KEY_ID", "secret_access_key": "AWS_SECRET_ACCESS_KEY"},
    )

    fe = feature_engineering_component(
        input_results_dataset=pull.outputs["state_output"],
        user_prompt=user_prompt,
    )

    pre = preprocessing_component(
        image_dir=pull.outputs["state_output"],
        img_size=img_size,
        batch_size=batch_size,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )
    # maintain notebook ordering: FE before preprocessing
    pre.after(fe)

    trn = train_model_component(
        manifest_path=pre.outputs["manifest_output"],
        model_id=model_id,
        epochs=epochs,
        lr=lr,
        fraction=train_fraction,
        batch_size=batch_size,
    )
    trn.set_cpu_request("12").set_memory_request("32Gi").set_accelerator_type("nvidia.com/gpu").set_accelerator_limit("1")

    evaluation_component(
        manifest_path=pre.outputs["manifest_output"],
        checkpoint_path=trn.outputs["checkpoint_output"],
        model_id=model_id,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        sample_count=sample_count,
    )


def compile_pipeline() -> None:
    compiler.Compiler().compile(pokegen_pipeline, str(PIPELINE_YAML))
    print(f"Compiled → {PIPELINE_YAML}")


def upload_and_run(host: str, *, run: bool = False) -> None:
    client = Client(host=host)

    # Upload (or create a new version if the pipeline already exists)
    pipeline = client.upload_pipeline(
        pipeline_package_path=str(PIPELINE_YAML),
        pipeline_name="pokegen-mlops",
    )
    print(f"Pipeline uploaded: {pipeline.pipeline_id}")

    if run:
        run_result = client.create_run_from_pipeline_package(
            pipeline_file=str(PIPELINE_YAML),
            arguments={},
            run_name="pokegen-mlops-run",
        )
        print(f"Run created: {run_result.run_id}")
        print(f"Track it at: {host}/#/runs/details/{run_result.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile and optionally upload the pokegen KFP pipeline.")
    parser.add_argument("--host", default=None, help="KFP host URL (e.g. https://kubeflow.example.com)")
    parser.add_argument("--run", action="store_true", help="Trigger an immediate run after uploading")
    args = parser.parse_args()

    compile_pipeline()

    if args.host:
        upload_and_run(args.host, run=args.run)
