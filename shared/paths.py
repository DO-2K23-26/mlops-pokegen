"""Chemins canoniques relatifs à la racine du dépôt pokegen."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CARDS_DIR = REPO_ROOT / "cards"
MLOPS_DIR = REPO_ROOT / "mlops"
CONFIGS_DIR = MLOPS_DIR / "configs"

# Données brutes synchronisées depuis S3 (DVC) — voir 00_pull_data_dvc.ipynb
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_CARDS_DIR = DATA_RAW_DIR / "cards"

# Dossiers d'étapes (ordre pipeline)
DIR_00_PULL_DATA = MLOPS_DIR / "00_pull_data"
DIR_01_FEATURE_ENGINEERING = MLOPS_DIR / "01_feature_engineering"
DIR_02_PREPROCESSING = MLOPS_DIR / "02_preprocessing"
DIR_03_TRAIN_MODEL = MLOPS_DIR / "03_train_model"
DIR_04_EVALUATION = MLOPS_DIR / "04_evaluation"
DIR_05_MODEL_REGISTRY = MLOPS_DIR / "05_model_registry"
DIR_06_DEPLOYMENT = MLOPS_DIR / "06_deployment"
DIR_07_MONITORING = MLOPS_DIR / "07_monitoring"
ARTIFACTS_DIR = MLOPS_DIR / "artifacts"
PIPELINE_ARTIFACTS_DIR = ARTIFACTS_DIR / "pipeline"
STEPS_ARTIFACTS_DIR = PIPELINE_ARTIFACTS_DIR / "steps"
PIPELINES_DIR = MLOPS_DIR / "pipelines"

# Fichiers lourds par étape (référencés dans steps/*/state.json)
STEP_02_MANIFEST = STEPS_ARTIFACTS_DIR / "02_preprocessing" / "preprocessing_manifest.json"
STEP_03_CHECKPOINT = STEPS_ARTIFACTS_DIR / "03_train_model" / "checkpoint.pt"
STEP_04_METRICS = STEPS_ARTIFACTS_DIR / "04_evaluation" / "metrics.json"
STEP_04_LOSS_PNG = STEPS_ARTIFACTS_DIR / "04_evaluation" / "loss_curves.png"
STEP_04_SAMPLES_PNG = STEPS_ARTIFACTS_DIR / "04_evaluation" / "eval_samples.png"
STEP_05_LORA_DIR = STEPS_ARTIFACTS_DIR / "05_model_registry" / "pokemon_card_lora"
STEP_05_HISTORY = STEP_05_LORA_DIR / "training_history.pt"
STEP_06_SAMPLES_DIR = STEPS_ARTIFACTS_DIR / "06_deployment" / "deployment_samples"
STEP_07_REPORT = STEPS_ARTIFACTS_DIR / "07_monitoring" / "monitoring_report.json"

# Rétrocompatibilité (anciens chemins plats — préférer STEP_* ci-dessus)
TRAIN_CHECKPOINT_DIR = STEPS_ARTIFACTS_DIR / "03_train_model"
TRAIN_CHECKPOINT_PATH = STEP_03_CHECKPOINT
MODEL_REGISTRY_LORA_DIR = STEP_05_LORA_DIR
