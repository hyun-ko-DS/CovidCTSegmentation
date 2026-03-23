import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_multi_samples(images, masks, num_samples=3, dataset_name="Dataset"):
    # GG(0번) 또는 Consol(1번) 병변이 있는 인덱스만 추출
    # masks의 값을 0.5 기준으로 이진화하여 노이즈 방지
    has_lesion = np.where(np.any(masks[:, :, :, :2] > 0.5, axis=(1, 2, 3)))[0]

    if len(has_lesion) < num_samples:
        indices = np.random.choice(len(images), num_samples, replace=False)
    else:
        indices = np.random.choice(has_lesion, num_samples, replace=False)

    for idx in indices:
        img = images[idx].squeeze()
        mask_gg = masks[idx, :, :, 0] > 0.5    # Boolean Mask for GG
        mask_consol = masks[idx, :, :, 1] > 0.5 # Boolean Mask for Consol

        # --- RGBA 오버레이 생성 (가장 확실한 방법) ---
        # 1. Ground Glass용 빨간색 레이어 (R=1, G=0, B=0, A=0.5)
        overlay_gg = np.zeros((*img.shape, 4))
        overlay_gg[mask_gg] = [1, 0, 0, 0.4] # 빨강, 투명도 0.4

        # 2. Consolidation용 초록색 레이어 (R=0, G=1, B=0, A=0.5)
        overlay_consol = np.zeros((*img.shape, 4))
        overlay_consol[mask_consol] = [0, 1, 0, 0.4] # 초록, 투명도 0.4

        plt.figure(figsize=(22, 5))
        plt.suptitle(f"[{dataset_name}] Sample Index: {idx}", fontsize=18, y=1.1)

        # 1. 원본 CT
        plt.subplot(1, 4, 1)
        plt.imshow(img, cmap='bone')
        plt.title("Original CT")
        plt.axis('off')

        # 2. Ground Glass (Red)
        plt.subplot(1, 4, 2)
        plt.imshow(img, cmap='bone')
        plt.imshow(overlay_gg) # RGBA 이미지는 cmap 없이 바로 표시됨
        plt.title("Ground Glass (Red)")
        plt.axis('off')

        # 3. Consolidation (Green)
        plt.subplot(1, 4, 3)
        plt.imshow(img, cmap='bone')
        plt.imshow(overlay_consol)
        plt.title("Consolidation (Green)")
        plt.axis('off')

        # 4. Total Overlay (Red + Green)
        plt.subplot(1, 4, 4)
        plt.imshow(img, cmap='bone')
        plt.imshow(overlay_gg)
        plt.imshow(overlay_consol)
        plt.title("Total Overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

def visualize_aug_comparison(images, masks, transform, num_samples=3):
    """
    images: (N, 512, 512, 1) 형태의 넘파이 배열 (float32, 0~1)
    masks: (N, 512, 512, 4) 형태의 넘파이 배열 (uint8 또는 float32, 0 또는 1)
    transform: Albumentations 변환 객체
    num_samples: 비교할 샘플 개수
    """
    # 1. 샘플링 로직 개선: GG(0번) 또는 Consol(1번) 병변이 하나라도 있는 이미지 선택
    # 마스크 값이 소수점일 경우를 대비해 0.5 기준으로 이진화하여 판단
    mask_check = masks[..., :2] > 0.5
    has_lesion = np.where(np.any(mask_check, axis=(1, 2, 3)))[0]

    if len(has_lesion) == 0:
        print("표시할 병변이 있는 이미지가 없습니다.")
        return

    indices = np.random.choice(has_lesion, num_samples, replace=False)

    # 색상 정의 (RGBA: 0~1 range)
    RED = [1, 0, 0, 0.4]    # Ground Glass: 빨강, 투명도 0.4
    GREEN = [0, 1, 0, 0.4]  # Consolidation: 초록, 투명도 0.4

    for idx in indices:
        orig_img = images[idx] # (512, 512, 1)
        orig_mask = masks[idx] # (512, 512, 4)

        # 2. 증강 적용
        augmented = transform(image=orig_img, mask=orig_mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']

        # 시각화를 위한 squeeze (512, 512, 1) -> (512, 512)
        orig_img_2d = orig_img.squeeze()
        aug_img_2d = aug_img.squeeze()

        # 3. RGBA 오버레이 생성 함수 (코드 중복 방지)
        def create_rgba_overlay(mask, h, w):
            # GG (Ch 0) 빨간색 레이어
            overlay_gg = np.zeros((h, w, 4))
            overlay_gg[mask[..., 0] > 0.5] = RED

            # Consol (Ch 1) 초록색 레이어
            overlay_consol = np.zeros((h, w, 4))
            overlay_consol[mask[..., 1] > 0.5] = GREEN

            return overlay_gg, overlay_consol

        h, w = orig_img_2d.shape
        orig_gg_layer, orig_consol_layer = create_rgba_overlay(orig_mask, h, w)
        aug_gg_layer, aug_consol_layer = create_rgba_overlay(aug_mask, h, w)

        # 4. 시각화 (1x2 Subplots)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # --- 왼쪽: 원본 ---
        axes[0].imshow(orig_img_2d, cmap='bone') # 베이스 CT
        axes[0].imshow(orig_gg_layer)           # GG 빨강 오버레이
        axes[0].imshow(orig_consol_layer)       # Consol 초록 오버레이
        axes[0].set_title(f"Original (Idx: {idx})\nRED: GG, GREEN: Consol", fontsize=12)
        axes[0].axis('off')

        # --- 오른쪽: 증강 결과 ---
        axes[1].imshow(aug_img_2d, cmap='bone') # 증강된 CT
        axes[1].imshow(aug_gg_layer)           # 증강된 GG 빨강 오버레이
        axes[1].imshow(aug_consol_layer)       # 증강된 Consol 초록 오버레이
        axes[1].set_title("Augmented Result\nRED: GG, GREEN: Consol", fontsize=12)
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

def calculate_pixel_ratio(masks):
    # 각 채널별로 1인 픽셀의 비중 계산
    total_pixels = masks.shape[0] * 512 * 512
    ratios = np.sum(masks, axis=(0, 1, 2)) / total_pixels * 100
    labels = ['GG', 'Consol', 'Other', 'BG']
    for label, ratio in zip(labels, ratios):
        print(f"{label}: {ratio:.4f}%")

def print_stats(name, data):
    print(f"[{name}]")
    print(f"  - Shape: {data.shape}")
    print(f"  - Dtype: {data.dtype}")
    print(f"  - Range: ({data.min():.2f}, {data.max():.2f})")
    print(f"  - Mean : {data.mean():.2f}")
    print("-" * 30)

def summarize_npy_files(path):
    summary = []
    for file in sorted(os.listdir(path)):
        if file.endswith('.npy'):
            data = np.load(os.path.join(path, file))

            # 1. 데이터의 첫 번째 차원이 '장수(Count)'입니다.
            count = data.shape[0]

            # 2. 파일명에 따른 데이터 유형 구분 (선택 사항)
            data_type = 'Mask' if 'mask' in file.lower() else 'Image'

            summary.append({
                'File Name': file,
                'Type': data_type,      # 이미지/마스크 구분
                'Count': count,         # 데이터 장수 추가
                'Shape': data.shape,
                'Dtype': data.dtype,
                'Min': f"{data.min():.2f}",
                'Max': f"{data.max():.2f}",
                'Mean': f"{data.mean():.2f}"
            })
    return pd.DataFrame(summary)

def analyze_class_distribution(masks, name="Dataset"):
    # masks shape: (N, 512, 512, 4)
    total_pixels = masks.shape[0] * 512 * 512

    classes = ['Ground Glass', 'Consolidation', 'Other Lungs', 'Background']
    dist = {}

    for i, cls_name in enumerate(classes):
        pixel_count = np.sum(masks[..., i])
        percentage = (pixel_count / total_pixels) * 100
        dist[cls_name] = percentage

    print(f"--- {name} Class Distribution (%) ---")
    for k, v in dist.items():
        print(f"{k}: {v:.4f}%")
    return dist

def plot_medseg_samples(images, masks, num_samples=3):
    """
    images: (N, 512, 512, 1) 또는 (N, 512, 512)
    masks: (N, 512, 512, 4)
    num_samples: 보여줄 샘플의 개수
    """
    # 1. 인덱스 무작위 선택
    num_total = len(images)
    num_samples = min(num_samples, num_total)
    selected_indices = np.random.choice(range(num_total), num_samples, replace=False)

    # 2. 시각화 루프
    for idx in selected_indices:
        # 1개의 원본 + 4개의 마스크 채널 = 총 5개 서브플롯
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        # [0] 원본 CT 이미지 (가장 왼쪽)
        img = images[idx].squeeze() # (512, 512, 1) -> (512, 512)
        axes[0].imshow(img, cmap='bone')
        axes[0].set_title(f"Medseg Original (Idx: {idx})")
        axes[0].axis('off')

        # [1-4] 마스크 채널들
        titles = ['0: Ground Glass', '1: Consolidation', '2: Other Lungs', '3: Background']
        for i in range(4):
            # 마스크는 'jet'이나 'hot' 컬러맵을 쓰면 강도가 더 뚜렷하게 보입니다
            axes[i+1].imshow(masks[idx, :, :, i], cmap='jet')
            axes[i+1].set_title(titles[i])
            axes[i+1].axis('off')

        plt.tight_layout()
        plt.show()

def plot_radiopedia_samples(images, masks, num_samples=3):
    """
    images: (N, 512, 512, 1) 또는 (N, 512, 512)
    masks: (N, 512, 512, 4)
    num_samples: 보여줄 샘플의 개수
    """
    # 1. 병변(GG 또는 Consolidation)이 있는 슬라이스 인덱스들만 추출
    has_lesion_indices = np.where(np.any(masks[:, :, :, :2] == 1, axis=(1, 2, 3)))[0]

    if len(has_lesion_indices) == 0:
        print("병변이 있는 슬라이스를 찾지 못했습니다.")
        return

    # 2. 요청한 샘플 수가 실제 병변 슬라이스 수보다 많으면 조정
    num_samples = min(num_samples, len(has_lesion_indices))

    # 3. 무작위로 샘플 인덱스 선택 (중복 없이)
    selected_indices = np.random.choice(has_lesion_indices, num_samples, replace=False)

    # 4. 시각화 시작
    for idx in selected_indices:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        # Original CT
        img = images[idx].squeeze()
        axes[0].imshow(img, cmap='bone')
        axes[0].set_title(f"Original CT (Idx: {idx})")
        axes[0].axis('off')

        # Mask Channels
        titles = ['0: Ground Glass', '1: Consolidation', '2: Other Lungs', '3: Background']
        for i in range(4):
            # 마스크가 잘 보이도록 배경은 검정, 마스크는 유색으로 표현
            axes[i+1].imshow(masks[idx, :, :, i], cmap='jet')
            axes[i+1].set_title(titles[i])
            axes[i+1].axis('off')

        plt.tight_layout()
        plt.show()

def calculate_pixel_ratio(masks):
    # 각 채널별로 1인 픽셀의 비중 계산
    total_pixels = masks.shape[0] * 512 * 512
    ratios = np.sum(masks, axis=(0, 1, 2)) / total_pixels * 100
    labels = ['GG', 'Consol', 'Other', 'BG']
    for label, ratio in zip(labels, ratios):
        print(f"{label}: {ratio:.4f}%")

def visualize_aug_comparison(images, masks, transform, num_samples=3):
    """
    images: (N, 512, 512, 1) 형태의 넘파이 배열 (float32, 0~1)
    masks: (N, 512, 512, 4) 형태의 넘파이 배열 (uint8 또는 float32, 0 또는 1)
    transform: Albumentations 변환 객체
    num_samples: 비교할 샘플 개수
    """
    # 1. 샘플링 로직 개선: GG(0번) 또는 Consol(1번) 병변이 하나라도 있는 이미지 선택
    # 마스크 값이 소수점일 경우를 대비해 0.5 기준으로 이진화하여 판단
    mask_check = masks[..., :2] > 0.5
    has_lesion = np.where(np.any(mask_check, axis=(1, 2, 3)))[0]

    if len(has_lesion) == 0:
        print("표시할 병변이 있는 이미지가 없습니다.")
        return

    indices = np.random.choice(has_lesion, num_samples, replace=False)

    # 색상 정의 (RGBA: 0~1 range)
    RED = [1, 0, 0, 0.4]    # Ground Glass: 빨강, 투명도 0.4
    GREEN = [0, 1, 0, 0.4]  # Consolidation: 초록, 투명도 0.4

    for idx in indices:
        orig_img = images[idx] # (512, 512, 1)
        orig_mask = masks[idx] # (512, 512, 4)

        # 2. 증강 적용
        augmented = transform(image=orig_img, mask=orig_mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']

        # 시각화를 위한 squeeze (512, 512, 1) -> (512, 512)
        orig_img_2d = orig_img.squeeze()
        aug_img_2d = aug_img.squeeze()

        # 3. RGBA 오버레이 생성 함수 (코드 중복 방지)
        def create_rgba_overlay(mask, h, w):
            # GG (Ch 0) 빨간색 레이어
            overlay_gg = np.zeros((h, w, 4))
            overlay_gg[mask[..., 0] > 0.5] = RED

            # Consol (Ch 1) 초록색 레이어
            overlay_consol = np.zeros((h, w, 4))
            overlay_consol[mask[..., 1] > 0.5] = GREEN

            return overlay_gg, overlay_consol

        h, w = orig_img_2d.shape
        orig_gg_layer, orig_consol_layer = create_rgba_overlay(orig_mask, h, w)
        aug_gg_layer, aug_consol_layer = create_rgba_overlay(aug_mask, h, w)

        # 4. 시각화 (1x2 Subplots)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # --- 왼쪽: 원본 ---
        axes[0].imshow(orig_img_2d, cmap='bone') # 베이스 CT
        axes[0].imshow(orig_gg_layer)           # GG 빨강 오버레이
        axes[0].imshow(orig_consol_layer)       # Consol 초록 오버레이
        axes[0].set_title(f"Original (Idx: {idx})\nRED: GG, GREEN: Consol", fontsize=12)
        axes[0].axis('off')

        # --- 오른쪽: 증강 결과 ---
        axes[1].imshow(aug_img_2d, cmap='bone') # 증강된 CT
        axes[1].imshow(aug_gg_layer)           # 증강된 GG 빨강 오버레이
        axes[1].imshow(aug_consol_layer)       # 증강된 Consol 초록 오버레이
        axes[1].set_title("Augmented Result\nRED: GG, GREEN: Consol", fontsize=12)
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()