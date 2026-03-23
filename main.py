import os
from pathlib import Path
import json
# import eda, msa
from loader import run_loading_pipeline
from train import run_training_pipeline

def load_config(config_path="config.json"):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    global BASE_PATH, RUN_NAME, SAVE_DIR, config

    BASE_PATH = "./results"
    RUN_NAME = 'exp_25_best_model'
    SAVE_DIR = os.path.join(BASE_PATH, RUN_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)
    config = load_config("config.json")

    run_loading_pipeline(login_wandb=True)

    run_training_pipeline(
        path=BASE_PATH,
        config_path="config.json",
        base_path="./results",
        run_name="exp_final_best_model",
    )