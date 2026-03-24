import json
from pathlib import Path
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def load_config(config_path="config.json"):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

class MSABlock(nn.Module):
    def __init__(self, in_channels, alpha=config['msa_alpha']):
        super().__init__()

        # --- 0. [Residual Scaling] Skip Connection 강도 설정 ---
        # 목적: 인코더의 오리지널 정보를 보존하면서 MSA의 보정치만 효과적으로 섞음.
        self.alpha = alpha

        # --- 1. [안전 장치] 채널 배분 로직 ---
        # 인코더에서 넘어온 대량의 채널을 4개의 경로(Path)로 분산하여 연산 효율을 높임.
        base_ch = in_channels // 4
        rem = in_channels % 4 # 4로 나누어 떨어지지 않을 때 남는 채널(나머지) 처리

        ch1 = base_ch + (1 if rem > 0 else 0)
        ch2 = base_ch + (1 if rem > 1 else 0)
        ch3 = base_ch + (1 if rem > 2 else 0)
        ch4 = in_channels - (ch1 + ch2 + ch3)

        active_channels = [c for c in [ch1, ch2, ch3, ch4] if c > 0]

        # --- 2. [Spatial Path] Multi-Scale Spatial Attention (공간 정보 강화) ---
        # 역할: 인코더의 Spatial Info를 다양한 Dilation으로 관찰.
        dilations = config['dilation_rates'] # Receptive Field를 단계적으로 확장

        self.spatial_scales = nn.ModuleList()
        # ch1: 1x1 Conv로 아주 미세한 점이나 선 포착
        if ch1 > 0:
            self.c.append(nn.Conv2d(in_channels, ch1, kernel_size=1))

        # Dilated Conv: 필터 사이의 간격을 벌려 넓은 맥락과 좁은 맥락의 공간 정보를 모두 추출
        for i, ch in enumerate(active_channels[1:], 1): # ch1 이후, 즉 ch2, ch3, ch4가 대상.
            d = dilations[i] # Dilation Rate

            # ch2 = 3 x 3 / ch3 = 5 x 5 / ch4 = 7 x 7
            self.spatial_scales.append(
                nn.Conv2d(in_channels, ch, kernel_size=3, padding=d, dilation=d)
            )

        # --- 3. [Semantic Path] Channel Attention (의미 정보 정제) ---
        # 역할: 다운샘플링-업샘플링을 거쳐 생성된 특징들 중 "병변과 관련된 핵심 의미"를 선택함.
        # SE Block: 노이즈 채널은 끄고, 진짜 병변(Semantic Info) 채널의 볼륨만 키움.
        # 앞 단계인 Spatial Path와 병렬적으로 작동.
        # 인코더의 feature map들이 들어오자마자 하나는 spatial path로, 다른 하나는 semantic path로.

        reduction = max(1, in_channels // 16) # reduction ratio = 16
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Squeeze: 전체 이미지의 의미를 1x1로 요약
            nn.Conv2d(in_channels, reduction, 1), # 압축을 통해 핵심 특징만 남김
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, in_channels, 1), # Excitation: 다시 원래 채널로 복원하며 중요도 부여
            nn.Sigmoid() # 0~1 사이의 가중치(Attention Map) 생성
        )

        # --- 4. 융합 (Final Fusion) ---
        # 정제된 공간 정보(Spatial)와 의미 정보(Semantic)를 하나로 통합하여 디코더로 전달.
        # 1 x 1 사용 이유 1) 공간 정보는 건드리지 않고, 채널들끼리만 정보를 주고받게 하기 위함.
        # 1 x 1 사용 이유 2) 채널 수가 변하지 않도록 하기 위함.
        self.fuse = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # [A] Spatial Branch: spatial path를 통해 나온 채널 결합: ch1 + ch2 + ch3 + ch4
        # 인코더의 세밀한 엣지 정보가 다양한 크기로 정제되어 모임.
        spatial_feats = torch.cat([layer(x) for layer in self.spatial_scales], dim=1)

        # [B] Channel Branch: SE-Block을 통해 나온 가중치를 인코더의 출력 (= MSA Block의 입력)에 곱해줌.
        # 불필요한 노이즈 신호를 걸러내는 필터링 과정.
        channel_feats = x * self.channel_attn(x)

        # [C] Fusion: "어디에 있는가(Spatial)"와 "어떤 의미인가(Semantic)"를 덧셈으로 융합 후 Conv로 섞음.
        # fuse:섞은 후에 1 x 1 conv 통과
        msa_out = self.fuse(spatial_feats + channel_feats)

        # [D] Skip Connection (Residual): 원본 x(인코더 정보) + (alpha * 정제된 정보).
        # x: 인코더를 통해 나온 출력
        # alpha * msa_out: 인코더를 통해 나온 출력을 MSA Block 통해 나온 값
        return x + self.alpha * msa_out

class MSASkipUnet(nn.Module):
    def __init__(self, config): # config를 인자로 받습니다.
        super().__init__()
        
        # UNET 모델 선언
        self.base_model = smp.Unet(
            encoder_name=config["encoder_name"],
            encoder_weights=config["encoder_weights"],
            in_channels=config["in_channels"],
            classes=config["classes"],
            encoder_depth=config["encoder_depth"],                 
            activation=None
        )

        # 인코더 출력 채널 수 (efficientnet-b4): 1 -> 48 -> 32 -> 56 -> 160 -> 448
        encoder_channels = self.base_model.encoder.out_channels 
        self.msa_blocks = nn.ModuleList([
            MSABlock(ch) for ch in encoder_channels
        ])

    def forward(self, x):
        # 인코더를 통해 나온 feature map들
        features = self.base_model.encoder(x) 

        # 각 층마다 MSA Block 적용
        msa_features = []
        for i, feat in enumerate(features):
            msa_feat = self.msa_blocks[i](feat) 
            msa_features.append(msa_feat)

        # MSA 를 거쳐 나온 정제된 feautre map들을 디코더로 전달
        decoder_output = self.base_model.decoder(msa_features)

        # 세그멘테잇녀 헤드로 최종 마스크 생성
        masks = self.base_model.segmentation_head(decoder_output)
        return masks