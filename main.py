# [ main.py ]
import os
from pathlib import Path
import json
import numpy as np
from loader import run_loading_pipeline
from train import run_training_pipeline, apply_lung_window 
from inference import run_inference_pipeline

def load_config(config_path="config.json"):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    # 설정 로드
    config = load_config("config.json")
    
    # 공통 설정 업데이트 (inference에서 쓰일 경로 정보)
    config["base_path"] = "./results"
    config["run_name"] = "exp_final_best_model"
    SAVE_DIR = os.path.join(config["base_path"], config["run_name"])
    os.makedirs(SAVE_DIR, exist_ok=True)