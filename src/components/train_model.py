"""KFP component — submits a PyTorchJob for DDP training and polls for completion."""

from kfp.dsl import component as kfp_component, Input, Output, Artifact

_GIT_PKG = "pokegen-shared @ git+https://github.com/DO-2K23-26/mlops-pokegen.git"


@kfp_component(packages_to_install=["kubernetes"])
def train_model_component(
    checkpoint_output: Output[Artifact],
    state_output: Output[Artifact],
    manifest_path: Input[Artifact],
    model_id: str = "runwayml/stable-diffusion-v1-5",
    epochs: int = 1,
    lr: float = 1e-5,
    fraction: float = 1.0,
    seed: int = 42,
    batch_size: int = 4,
    num_workers: int = 4,
    cards_pvc_name: str = "pokegen-data",
    namespace: str = "pokegen",
    num_nodes: int = 2,
) -> None:
    """Submit a PyTorchJob for DDP training across GPU nodes and poll for completion."""
    import json
    import time
    import uuid
    from pathlib import Path

    from kubernetes import client as k8s_client, config as k8s_config

    k8s_config.load_incluster_config()
    custom_api = k8s_client.CustomObjectsApi()

    # Stage manifest onto the PVC so PyTorchJob pods can access it
    manifest_pvc_path = "/data/manifests/train_manifest.json"
    p = Path(manifest_pvc_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(Path(manifest_path.path).read_text(encoding="utf-8"), encoding="utf-8")

    checkpoint_pvc_path = "/data/checkpoints/lora_checkpoint.pt"

    _GIT_PKG = "pokegen-shared @ git+https://github.com/DO-2K23-26/mlops-pokegen.git"
    _TRAIN_IMAGE = "pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime"

    train_cmd = " ".join([
        f"pip install --quiet '{_GIT_PKG}' &&",
        "python -m shared.train_distributed",
        f"--manifest-path {manifest_pvc_path}",
        f"--checkpoint-path {checkpoint_pvc_path}",
        f"--model-id {model_id}",
        f"--epochs {epochs}",
        f"--lr {lr}",
        f"--fraction {fraction}",
        f"--seed {seed}",
        f"--batch-size {batch_size}",
        f"--num-workers {num_workers}",
    ])

    job_name = f"pokegen-train-{uuid.uuid4().hex[:8]}"

    def _replica_spec(replicas: int) -> dict:
        return {
            "replicas": replicas,
            "restartPolicy": "OnFailure",
            "template": {
                "spec": {
                    "tolerations": [{
                        "key": "nvidia.com/gpu",
                        "operator": "Equal",
                        "value": "present",
                        "effect": "NoSchedule",
                    }],
                    "volumes": [{"name": "data", "persistentVolumeClaim": {"claimName": cards_pvc_name}}],
                    "containers": [{
                        "name": "pytorch",
                        "image": _TRAIN_IMAGE,
                        "command": ["bash", "-c"],
                        "args": [train_cmd],
                        "resources": {
                            "requests": {"nvidia.com/gpu": "1", "cpu": "12", "memory": "32Gi"},
                            "limits": {"nvidia.com/gpu": "1"},
                        },
                        "volumeMounts": [{"name": "data", "mountPath": "/data"}],
                    }],
                }
            },
        }

    replica_specs: dict = {"Master": _replica_spec(1)}
    if num_nodes > 1:
        replica_specs["Worker"] = _replica_spec(num_nodes - 1)

    pytorchjob = {
        "apiVersion": "kubeflow.org/v1",
        "kind": "PyTorchJob",
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {"pytorchReplicaSpecs": replica_specs},
    }

    custom_api.create_namespaced_custom_object(
        group="kubeflow.org", version="v1",
        namespace=namespace, plural="pytorchjobs", body=pytorchjob,
    )
    print(f"PyTorchJob '{job_name}' submitted in namespace '{namespace}'")

    while True:
        job = custom_api.get_namespaced_custom_object(
            group="kubeflow.org", version="v1",
            namespace=namespace, plural="pytorchjobs", name=job_name,
        )
        phase = None
        for cond in job.get("status", {}).get("conditions", []):
            if cond.get("status") != "True":
                continue
            if cond.get("type") == "Succeeded":
                phase = "Succeeded"
            elif cond.get("type") == "Failed":
                phase = "Failed"

        if phase == "Succeeded":
            print("PyTorchJob succeeded")
            break
        if phase == "Failed":
            raise RuntimeError(f"PyTorchJob '{job_name}' failed")

        time.sleep(30)

    # Write checkpoint pointer so evaluation_component can find the file on the PVC
    ckpt_out = Path(checkpoint_output.path)
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_out.write_text(json.dumps({"checkpoint_path": checkpoint_pvc_path}), encoding="utf-8")

    # Read losses written by rank-0 during training
    results_path = Path(checkpoint_pvc_path).parent / "train_results.json"
    try:
        results = json.loads(results_path.read_text())
        train_losses = results.get("train_losses", [])
        val_losses = results.get("val_losses", [])
    except Exception:
        train_losses, val_losses = [], []

    out = Path(state_output.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "step": "03_train_model",
        "job_name": job_name,
        "manifest_path": manifest_pvc_path,
        "checkpoint_path": checkpoint_pvc_path,
        "model_id": model_id,
        "epochs_done": len(train_losses),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
