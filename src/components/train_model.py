"""Component for notebook 03_train_model.ipynb."""

from kfp.dsl import component as kfp_component, Input, Output, Artifact

_GIT_PKG = "pokegen-shared @ git+https://github.com/DO-2K23-26/mlops-pokegen.git"


@kfp_component(packages_to_install=["torch", "transformers", "diffusers", "peft", "pillow", "tqdm", _GIT_PKG, "kubeflow"])
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
    num_workers: int = 0,
) -> None:
    """Train the LoRA U-Net, persist the checkpoint, and write step state."""
    from kubeflow.trainer import TrainerClient, CustomTrainer
    from shared.train_distributed import train_pytorch

    job_id = TrainerClient().train(
        trainer=CustomTrainer(
            func=train_pytorch,
            func_kwargs={
                "manifest_path": manifest_path.path,
                "checkpoint_path": checkpoint_output.path,
                "state_path": state_output.path,
                "model_id": model_id,
                "epochs": epochs,
                "lr": lr,
                "fraction": fraction,
                "seed": seed,
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
            num_nodes=4,
            resources_per_node={
                "cpu": 3,
                "memory": "16Gi",
                "gpu": 1,  # Comment this line if you don't have GPUs.
            },
        )
    )

    for s in TrainerClient().get_job(name=job_id).steps:
        print(f"Step: {s.name}, Status: {s.status}, Devices: {s.device} x {s.device_count}")

    for logline in TrainerClient().get_job_logs(job_id, follow=True):
        print(logline)
