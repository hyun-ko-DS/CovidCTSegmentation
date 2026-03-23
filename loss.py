import json
from pathlib import Path
import torch
import torch.nn as nn

def load_config(config_path="config.json"):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

class DiceFocalLoss(nn.Module):
    def __init__(self, gamma=config['gamma'], alpha=config['alpha'], 
                smooth=config['smooth'], class_weights=config['class_weights']):
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma      # 어려운 샘플에 집중하는 정도
        self.alpha = alpha      # 클래스 불균형 조절 가중치
        self.smooth = smooth
        self.weights = weights  # 클래스별 중요도 가중치 (리스트 또는 텐서)

    def forward(self, inputs, targets):
        # 1. Softmax를 통한 확률값 변환 (Multi-class 채널 경쟁)
        probs = torch.softmax(inputs, dim=1)

        num_classes = inputs.shape[1]
        total_loss = 0

        # 각 클래스(채널)별로 개별 손실을 계산하여 합산
        for i in range(num_classes):
            # 채널별 추출 및 Flatten (픽셀 단위 연산을 위해 평탄화)
            probs_flat = probs[:, i, :, :].reshape(-1)
            targets_flat = targets[:, i, :, :].reshape(-1)

            # --- [Dice Loss 계산] ---
            intersection = (probs_flat * targets_flat).sum()
            dice_score = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
            dice_loss = 1 - dice_score

            # --- [Focal Loss 계산] ---
            # 수치적 안정성을 위해 확률값이 0이나 1이 되지 않도록 조정
            probs_flat = torch.clamp(probs_flat, min=1e-7, max=1-1e-7)

            # Binary Cross Entropy 계산
            BCE = - (targets_flat * torch.log(probs_flat) + (1 - targets_flat) * torch.log(1 - probs_flat))

            # Focal Weight 계산: (1 - pt)^gamma
            pt = torch.where(targets_flat == 1, probs_flat, 1 - probs_flat)
            focal_weight = (1 - pt) ** self.gamma

            # 최종 Focal Loss (해당 클래스의 평균값)
            focal_loss = (self.alpha * focal_weight * BCE).mean()

            # --- [클래스별 가중치 적용] ---
            class_loss = dice_loss + focal_loss
            if self.weights is not None:
                class_loss *= self.weights[i]

            total_loss += class_loss

        # 전과목 평균 점수 반환
        return total_loss / num_classes