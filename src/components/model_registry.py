"""Component for notebook 05_model_registry.ipynb."""

from kfp.dsl import component as kfp_component, Input, Output, Artifact


@kfp_component(packages_to_install=["model-registry==0.3.9"])
def model_registry_component(
    state_output: Output[Artifact],
    checkpoint_path: Input[Artifact],
    evaluation_state: Input[Artifact],
    model_name: str = "pokegen-lora-sd15",
    registry_host: str = "http://model-registry-service.default.svc.cluster.local",
    registry_port: int = 8080,
    author: str = "pokegen-pipeline",
    kubeflow_namespace: str = "pokegen",
) -> None:
    """Register the trained LoRA checkpoint in the Kubeflow Model Registry."""
    import json
    from pathlib import Path

    from model_registry import ModelRegistry

    # Resolve checkpoint path — two formats:
    # 1. JSON pointer {"checkpoint_path": "/data/..."} written by the PyTorchJob launcher
    # 2. Raw binary .pt file written by the single-GPU train_model_component
    _default_pvc_path = "/data/checkpoints/lora_checkpoint.pt"
    try:
        pointer = json.loads(Path(checkpoint_path.path).read_bytes().decode("utf-8"))
        checkpoint_pvc_path = pointer["checkpoint_path"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError):
        import shutil
        Path(_default_pvc_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path.path, _default_pvc_path)
        checkpoint_pvc_path = _default_pvc_path
        print(f"Copied binary checkpoint to PVC: {checkpoint_pvc_path}")

    # Read evaluation results for metadata
    eval_state = json.loads(Path(evaluation_state.path).read_text())
    final_val_loss = eval_state.get("final_val_loss")
    model_id = eval_state.get("model_id", "runwayml/stable-diffusion-v1-5")

    registry = ModelRegistry(
        server_address=registry_host,
        port=registry_port,
        author=author,
        is_secure=False,
    )

    # Semantic versioning: find the next vN by probing existing versions
    version = "v1"
    n = 1
    while True:
        try:
            registry.get_model_version(model_name, f"v{n}")
            n += 1
        except Exception:
            version = f"v{n}"
            break

    metadata: dict = {"base_model": model_id, "namespace": kubeflow_namespace}
    if final_val_loss is not None:
        metadata["val_loss"] = round(float(final_val_loss), 6)

    registry.register_model(
        model_name,
        f"pvc://{checkpoint_pvc_path}",
        model_format_name="pytorch",
        model_format_version="2",
        version=version,
        description=f"SD 1.5 LoRA fine-tuned on Pokémon cards — val_loss={final_val_loss:.4f}" if final_val_loss else "SD 1.5 LoRA fine-tuned on Pokémon cards",
        metadata=metadata,
    )

    # Retrieve IDs needed for KServe InferenceService labels
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
        "checkpoint_path": checkpoint_pvc_path,
        "val_loss": final_val_loss,
        "registry_host": registry_host,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
