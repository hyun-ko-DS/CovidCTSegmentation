import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

import kagglehub
import wandb

def run_loading_pipeline(
    kaggle_json_path="./kaggle.json",
    competition="covid-segmentation",
    login_wandb=True,
    local_data_dir="./data",
):
    if login_wandb:
        load_dotenv(".env")
        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            wandb.login(key=wandb_key)
            print("Wandb logged in successfully")
        else:
            print("WANDB_API_KEY not found. Skipping wandb login.")

    src_kaggle_json = Path(kaggle_json_path).expanduser().resolve()
    if not src_kaggle_json.exists():
        raise FileNotFoundError(f"kaggle.json not found: {src_kaggle_json}")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    dst_kaggle_json = kaggle_dir / "kaggle.json"

    shutil.copy2(src_kaggle_json, dst_kaggle_json)
    os.chmod(dst_kaggle_json, 0o600)
    print("Kaggle 인증 완료!")

    downloaded_path = Path(kagglehub.competition_download(competition)).resolve()

    local_path = Path(local_data_dir).expanduser().resolve()
    local_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(downloaded_path, local_path, dirs_exist_ok=True)

    print("\n캐시 다운로드 경로:", downloaded_path)
    print("프로젝트 데이터 경로:", local_path)
    print("파일 목록:", os.listdir(local_path))
    return str(local_path)