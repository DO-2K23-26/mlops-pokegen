"""Component for notebook 05_model_registry.ipynb."""

from kfp.dsl import component as kfp_component, Input, Output, Artifact


@kfp_component(packages_to_install=["model-registry==0.3.9", "boto3"])
def model_registry_component(
    state_output: Output[Artifact],
    checkpoint_path: Input[Artifact],
    evaluation_state: Input[Artifact],
    model_name: str = "pokegen-lora-sd15",
    r2_bucket: str = "mlops-pokemon",
    r2_endpoint_url: str = "https://c2feb305a6d15f2c6236f90b01a8e83a.r2.cloudflarestorage.com",
    registry_host: str = "http://model-registry-service.pokegen.svc.cluster.local",
    registry_port: int = 8080,
    author: str = "pokegen-pipeline",
    kubeflow_namespace: str = "pokegen",
) -> None:
    """Upload LoRA checkpoint to R2 and register it in the namespace Model Registry."""
    import json
    import os
    import shutil
    from pathlib import Path

    import boto3
    from model_registry import ModelRegistry
    from model_registry import utils as mr_utils

    # --- resolve checkpoint file ---
    # Handles two formats:
    #   1. JSON pointer {"checkpoint_path": "/data/..."} from PyTorchJob launcher
    #   2. Raw binary .pt file from single-GPU train_model_component
    _default_pvc_path = "/data/checkpoints/lora_checkpoint.pt"
    try:
        pointer = json.loads(Path(checkpoint_path.path).read_bytes().decode("utf-8"))
        checkpoint_pvc_path = pointer["checkpoint_path"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError):
        Path(_default_pvc_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path.path, _default_pvc_path)
        checkpoint_pvc_path = _default_pvc_path

    # --- read evaluation metadata ---
    eval_state = json.loads(Path(evaluation_state.path).read_text())
    final_val_loss = eval_state.get("final_val_loss")
    model_id = eval_state.get("model_id", "runwayml/stable-diffusion-v1-5")

    # --- semantic versioning: find next vN ---
    registry = ModelRegistry(
        server_address=registry_host,
        port=registry_port,
        author=author,
        is_secure=False,
    )
    version = "v1"
    n = 1
    while True:
        try:
            registry.get_model_version(model_name, f"v{n}")
            n += 1
        except Exception:
            version = f"v{n}"
            break

    # --- upload checkpoint to R2 ---
    s3_key = f"models/{model_name}/{version}/lora_checkpoint.pt"
    s3 = boto3.client(
        "s3",
        endpoint_url=r2_endpoint_url,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    s3.upload_file(checkpoint_pvc_path, r2_bucket, s3_key)
    print(f"Uploaded checkpoint to s3://{r2_bucket}/{s3_key}")

    # --- register model ---
    metadata: dict = {
        "base_model": model_id,
        "namespace": kubeflow_namespace,
    }
    if final_val_loss is not None:
        metadata["val_loss"] = round(float(final_val_loss), 6)

    registry.register_model(
        model_name,
        uri=mr_utils.s3_uri_from(s3_key, r2_bucket, endpoint=r2_endpoint_url),
        model_format_name="pytorch",
        model_format_version="2",
        version=version,
        description=(
            f"SD 1.5 LoRA fine-tuned on Pokémon cards — val_loss={final_val_loss:.4f}"
            if final_val_loss else "SD 1.5 LoRA fine-tuned on Pokémon cards"
        ),
        metadata=metadata,
    )

    registered_model = registry.get_registered_model(model_name)
    model_version = registry.get_model_version(model_name, version)
    print(f"Registered '{model_name}' {version} (model_id={registered_model.id}, version_id={model_version.id})")

    out = Path(state_output.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "step": "05_model_registry",
        "model_name": model_name,
        "version": version,
        "registered_model_id": registered_model.id,
        "model_version_id": model_version.id,
        "s3_uri": mr_utils.s3_uri_from(s3_key, r2_bucket, endpoint=r2_endpoint_url),
        "val_loss": final_val_loss,
        "registry_host": registry_host,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
