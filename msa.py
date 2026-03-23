import json
from pathlib import Path
import torch
import torch.nn as nn

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
        # alphadent 프로젝트의 핵심 전략을 이식하여 모델이 길을 잃지 않게 붙잡아주는 안전장치.
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
        # 역할: 인코더의 Spatial Info(위치 정보)를 다양한 크기의 눈(Dilation)으로 관찰함.
        # Dilation 1~3을 통해 '안개처럼 퍼진 GGO'와 '작고 단단한 침윤'의 위치를 동시에 포착.
        dilations = config['dilation_rates'] # 수용 영역(Receptive Field)을 단계적으로 확장

        self.spatial_scales = nn.ModuleList()
        # 1x1 Conv: 입력된 공간 특징을 선형적으로 압축/정제
        if ch1 > 0:
            self.spatial_scales.append(nn.Conv2d(in_channels, ch1, kernel_size=1))

        # Dilated Conv: 필터 사이의 간격을 벌려 넓은 맥락과 좁은 맥락의 공간 정보를 모두 추출
        for i, ch in enumerate(active_channels[1:], 1):
            d = dilations[i]
            self.spatial_scales.append(
                nn.Conv2d(in_channels, ch, kernel_size=3, padding=d, dilation=d)
            )

        # --- 3. [Semantic Path] Channel Attention (의미 정보 정제) ---
        # 역할: 다운샘플링-업샘플링을 거쳐 생성된 특징들 중 "병변과 관련된 핵심 의미"를 선택함.
        # SE Block 기법: 노이즈 채널은 끄고, 진짜 병변(Semantic Info) 채널의 볼륨만 키움.
        reduction = max(1, in_channels // 16)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Squeeze: 전체 이미지의 의미를 1x1로 요약
            nn.Conv2d(in_channels, reduction, 1), # 압축을 통해 핵심 특징만 남김
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, in_channels, 1), # Excitation: 다시 원래 채널로 복원하며 중요도 부여
            nn.Sigmoid() # 0~1 사이의 가중치(Attention Map) 생성
        )

        # --- 4. 융합 (Final Fusion) ---
        # 정제된 공간 정보(Spatial)와 의미 정보(Semantic)를 하나로 통합하여 데코더로 전달.
        self.fuse = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # [A] Spatial Branch: 돋보기와 망원경(Multi-Scale)으로 찾아낸 위치 정보들을 병합.
        # 인코더의 세밀한 엣지 정보가 다양한 크기로 정제되어 모임.
        spatial_feats = torch.cat([layer(x) for layer in self.spatial_scales], dim=1)

        # [B] Channel Branch: 특징 채널 중 "이것이 코로나 병변이다"라고 판단된 의미 정보에 가중치 곱.
        # 불필요한 노이즈 신호를 걸러내는 필터링 과정.
        channel_feats = x * self.channel_attn(x)

        # [C] Fusion: "어디에 있는가(Spatial)"와 "어떤 의미인가(Semantic)"를 덧셈으로 융합 후 Conv로 섞음.
        msa_out = self.fuse(spatial_feats + channel_feats)

        # [D] Skip Connection (Residual): 원본 x(인코더 정보) + (alpha * 정제된 정보).
        # x: 기존의 안정적인 신호 흐름 보존
        # alpha * msa_out: MSA가 찾아낸 새로운 힌트를 적절한 강도로 추가
        return x + self.alpha * msa_out