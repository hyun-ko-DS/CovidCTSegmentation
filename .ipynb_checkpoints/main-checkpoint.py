# [ main.py ]
import os
from pathlib import Path
import json
import numpy as np
from loader import run_loading_pipeline
from train import run_training_pipeline, apply_lung_window # 필요한 함수 import
from inference import run_inference_pipeline # 추가

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

    # 1. 데이터 로드 (WandB 로그인 포함)
    # run_loading_pipeline(login_wandb=True)

    # 2. 학습 실행 (데이터를 리턴받도록 train.py가 구성되어야 함)
    # print("🚀 학습 및 추론 파이프라인 시작")
    # best_trained_model = run_training_pipeline(
    #     path=None,
    #     config_path="config.json",
    #     base_path=config["base_path"],
    #     run_name=config["run_name"],
    # )

    # 3. 데이터 직접 로드 (Inference를 위해 필요)
    # train.py 내부의 로직을 참고하여 동일하게 구성
    images_medseg = np.load("data/images_medseg.npy")
    masks_medseg = np.load("data/masks_medseg.npy")
    images_radiopedia = np.load("data/images_radiopedia.npy")
    masks_radiopedia = np.load("data/masks_radiopedia.npy")
    test_images_medseg = np.load("data/test_images_medseg.npy")

    # 윈도잉 적용 및 데이터 분할 (train.py와 동일한 Seed 사용 필수)
    from sklearn.model_selection import train_test_split
    X_all = np.concatenate([apply_lung_window(images_medseg, config), apply_lung_window(images_radiopedia, config)], axis=0)
    Y_all = np.concatenate([masks_medseg, masks_radiopedia], axis=0).astype(np.float32)

    _, X_val, _, Y_val = train_test_split(
        X_all, Y_all, 
        test_size=config['validation_size'], 
        random_state=config['seed'], 
        shuffle=True
    )

    # 4. 추론 파이프라인 실행
    run_inference_pipeline(
        config=config, 
        X_val=X_val, 
        Y_val=Y_val, 
        test_images_medseg=test_images_medseg
    )