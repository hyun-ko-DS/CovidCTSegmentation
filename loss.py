import torch
import torch.nn as nn

class DiceFocalLoss(nn.Module):
    def __init__(self, config):
        super(DiceFocalLoss, self).__init__()
        self.gamma = config['gamma']
        self.alpha = config['alpha']
        self.smooth = config['smooth']
        self.weights = torch.tensor(config['class_weights'])

    def forward(self, inputs, targets):
        # 모델의 출력 (Logits)을 softmax를 통해 전체 합 1로 변환
        probs = torch.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        total_loss = 0

        for i in range(num_classes):
            probs_flat = probs[:, i, :, :].reshape(-1) # p_i = 해당 클래스의 예측 화률 (0 ~ 1)
            targets_flat = targets[:, i, :, :].reshape(-1) # t_i = 해당 클래스의 실제 정답 (0 or 1)

            # Dice Loss: 병변이 작더라도 전체 면적 대비 겸침 비율 파악 -> 영역 특화
            intersection = (probs_flat * targets_flat).sum()
            dice_score = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
            dice_loss = 1 - dice_score

            # Focal Loss: 난이도 기반 가중치
            probs_flat = torch.clamp(probs_flat, min=1e-7, max=1-1e-7) # log(0) 방지 (안전장치)
            # p_t: 모델이 정답을 맞힌 확률
            pt = torch.where(targets_flat > 0.5, probs_flat, 1 - probs_flat) 
            
            # BCE 직접 계산 시 안정성 확보
            BCE = -torch.log(pt) 
            focal_loss = (self.alpha * (1 - pt) ** self.gamma * BCE).mean()

            class_loss = dice_loss + focal_loss
            if self.weights is not None:
                class_loss *= self.weights[i]

            total_loss += class_loss

        return total_loss / num_classes