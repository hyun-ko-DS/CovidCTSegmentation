import json
import numpy as np
from pathlib import Path


def load_config(config_path="config.json"):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# HU Windowing : 0 ~ 1로 정규화
def apply_lung_window(images, config):
    images = images.astype(np.float32)
    window_min, window_max = config['window_min'], config['window_max']
    images = np.clip(images, window_min, window_max)
    return (images - window_min) / (window_max - window_min)
