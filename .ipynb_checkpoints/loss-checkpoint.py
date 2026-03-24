import torch
import torch.nn as nn

class DiceFocalLoss(nn.Module):
    def __init__(self, config): # config 딕셔너리를 직접 받도록 수정
        super(DiceFocalLoss, self).__init__()
        self.gamma = config['gamma']
        self.alpha = config['alpha']
        self.smooth = config['smooth']
        
        # 장치(device) 설정을 외부에서 받아 확실하게 지정
        # register_buffer를 쓰면 모델 이동 시 장치 할당이 자동화됩니다.
        self.register_buffer('weights', torch.tensor(config['class_weights']))

    def forward(self, inputs, targets):
        # inputs: [B, 4, H, W] (Logits)
        probs = torch.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        total_loss = 0

        for i in range(num_classes):
            probs_flat = probs[:, i, :, :].reshape(-1)
            targets_flat = targets[:, i, :, :].reshape(-1)

            # Dice Loss
            intersection = (probs_flat * targets_flat).sum()
            dice_score = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
            dice_loss = 1 - dice_score

            # Focal Loss (수치 안정성을 위해 targets > 0.5 사용)
            probs_flat = torch.clamp(probs_flat, min=1e-7, max=1-1e-7)
            # targets_flat == 1 대신 임계값을 사용해 float 연산 오차 방지
            pt = torch.where(targets_flat > 0.5, probs_flat, 1 - probs_flat)
            
            # BCE 직접 계산 시 안정성 확보
            BCE = -torch.log(pt) 
            focal_loss = (self.alpha * (1 - pt) ** self.gamma * BCE).mean()

            class_loss = dice_loss + focal_loss
            if self.weights is not None:
                class_loss *= self.weights[i]

            total_loss += class_loss

        return total_loss / num_classes