import os
from pathlib import Path
import json
import numpy as np

def load_config(config_path="config.json"):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_config(config, config_path="config.json"):
    """수정된 설정값을 JSON 파일로 저장합니다."""
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"📝 설정값이 {config_path}에 업데이트되었습니다.")

if __name__ == "__main__":
    # 설정 로드
    try:
        config_path = "config.json"
        config = load_config(config_path)
    except:
        print("config.json을 폴더에 업로드해주세요. 하이퍼파라미터 관련해서는 README 참고해주세요.")

    config["base_path"] = "./results"
    config["run_name"] = "exp_final"
    save_config(config, config_path)
    
    SAVE_DIR = os.path.join(config["base_path"], config["run_name"])
    os.makedirs(SAVE_DIR, exist_ok=True)