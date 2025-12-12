import os
import wandb
from loadotenv import load_env 

load_env()
# Local folder and filename for the downloaded model
MODELS_DIR = "../models"
MODEL_FILENAME = "best_model.pth"

os.makedirs(MODELS_DIR, exist_ok=True)

def download_artifact():
    assert 'WANDB_API_KEY' in os.environ, "WANDB_API_KEY not found in environment variables"
    wandb_org = os.environ.get("WANDB_ORG")
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_model_name = os.environ.get("WANDB_MODEL_NAME")
    wandb_model_version = os.environ.get("WANDB_MODEL_VERSION")

    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
    api = wandb.Api()

    #artifact_path = "username/project_name/artifact_name:version"
    artifact_path = "pensive1881/mlops_dsr_batch_44/resnet18:v0"
    artifact = api.artifact(artifact_path, type="model")
    artifact.download(root=MODELS_DIR)

download_artifact()
