"""KFP component — submits a TrainJob (Kubeflow Training v2) for DDP training and polls for completion."""

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
    """Submit a TrainJob (trainer.kubeflow.org/v1alpha1) for DDP training across GPU nodes and poll for completion."""
    import json
    import time
    import uuid
    from pathlib import Path

    import os
    from kubernetes import client as k8s_client

    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"

    print(f"[DEBUG] KUBERNETES_SERVICE_HOST={os.environ.get('KUBERNETES_SERVICE_HOST')}")
    print(f"[DEBUG] KUBERNETES_SERVICE_PORT={os.environ.get('KUBERNETES_SERVICE_PORT')}")
    print(f"[DEBUG] token exists={os.path.exists(token_path)}, ca exists={os.path.exists(ca_path)}")

    # Always use the standard in-cluster DNS name, bypassing KUBERNETES_SERVICE_HOST
    # which KFP launcher may set to the ml-pipeline server address.
    configuration = k8s_client.Configuration()
    configuration.host = "https://kubernetes.default.svc.cluster.local"
    configuration.verify_ssl = True
    configuration.ssl_ca_cert = ca_path
    with open(token_path) as f:
        token = f.read().strip()
    configuration.api_key = {"authorization": "Bearer " + token}
    print(f"[DEBUG] Using API host: {configuration.host}, token length: {len(token)}")
    custom_api = k8s_client.CustomObjectsApi(k8s_client.ApiClient(configuration))

    # Stage manifest onto the PVC so TrainJob pods can access it
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

    trainjob = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {
            "runtimeRef": {
                "name": "torch-distributed",
                "apiGroup": "trainer.kubeflow.org",
                "kind": "ClusterTrainingRuntime",
            },
            "trainer": {
                "image": _TRAIN_IMAGE,
                "command": ["bash", "-c"],
                "args": [train_cmd],
                "numNodes": num_nodes,
                "numProcPerNode": 1,
                "resourcesPerNode": {
                    "requests": {"nvidia.com/gpu": "1", "cpu": "12", "memory": "32Gi"},
                    "limits": {"nvidia.com/gpu": "1"},
                },
            },
            "runtimePatches": [
                {
                    "manager": "pokegen-pipeline",
                    "trainingRuntimeSpec": {
                        "template": {
                            "spec": {
                                "replicatedJobs": [
                                    {
                                        "name": "node",
                                        "template": {
                                            "spec": {
                                                "template": {
                                                    "spec": {
                                                        "tolerations": [
                                                            {
                                                                "key": "nvidia.com/gpu",
                                                                "operator": "Equal",
                                                                "value": "present",
                                                                "effect": "NoSchedule",
                                                            }
                                                        ],
                                                        "volumes": [
                                                            {
                                                                "name": "data",
                                                                "persistentVolumeClaim": {
                                                                    "claimName": cards_pvc_name
                                                                },
                                                            }
                                                        ],
                                                        "containers": [
                                                            {
                                                                "name": "node",
                                                                "volumeMounts": [
                                                                    {
                                                                        "name": "data",
                                                                        "mountPath": "/data",
                                                                    }
                                                                ],
                                                            }
                                                        ],
                                                    }
                                                }
                                            }
                                        },
                                    }
                                ]
                            }
                        }
                    },
                }
            ],
        },
    }

    custom_api.create_namespaced_custom_object(
        group="trainer.kubeflow.org", version="v1alpha1",
        namespace=namespace, plural="trainjobs", body=trainjob,
    )
    print(f"TrainJob '{job_name}' submitted in namespace '{namespace}'")

    while True:
        job = custom_api.get_namespaced_custom_object(
            group="trainer.kubeflow.org", version="v1alpha1",
            namespace=namespace, plural="trainjobs", name=job_name,
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
            print("TrainJob succeeded")
            break
        if phase == "Failed":
            raise RuntimeError(f"TrainJob '{job_name}' failed")

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
