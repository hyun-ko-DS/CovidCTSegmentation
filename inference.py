import os
import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score
from sklearn.model_selection import train_test_split

# 프로젝트 모듈 임포트
from loss import DiceFocalLoss
from msa import MSABlock, MSASkipUnet
from train import apply_lung_window
from utils import *

# ==========================================
# 1. 공통 유틸리티 함수 (Shared Helpers)
# ==========================================

def get_overlay_mask(mask, is_ch_first=True):
    """병변 마스크를 시각화용 RGBA 오버레이로 변환"""
    RED, GREEN = [1, 0, 0, 0.4], [0, 1, 0, 0.4]
    overlay_gg = np.zeros((512, 512, 4))
    overlay_cn = np.zeros((512, 512, 4))

    if is_ch_first:
        overlay_gg[mask[0] > 0.5] = RED
        overlay_cn[mask[1] > 0.5] = GREEN
    else:
        overlay_gg[mask[:, :, 0] > 0.5] = RED
        overlay_cn[mask[:, :, 1] > 0.5] = GREEN
    return overlay_gg, overlay_cn

def predict_probabilities(model, images, config):
    """이미지 리스트로부터 시그모이드 확률맵 추출 (Inference Engine)"""
    model.eval()
    all_probs = []
    device = config["device"]

    with torch.no_grad():
        for i in tqdm(range(len(images)), desc="Inference"):
            img_tensor = torch.from_numpy(images[i]).permute(2, 0, 1).unsqueeze(0).to(device).float()
            output = model(img_tensor)
            prob = torch.sigmoid(output).cpu().squeeze().numpy() 
            all_probs.append(prob)
            
    return np.array(all_probs)

# ==========================================
# 2. 검증(Validation) 전용 로직
# ==========================================

def find_best_threshold(probs, masks, config, save_path):
    """Macro F1 최대화 임계값 탐색"""
    y_prob_gg = np.concatenate([p[0].flatten() for p in probs])
    y_true_gg = np.concatenate([m[:, :, 0].flatten() for m in masks]) > 0.5
    y_prob_cs = np.concatenate([p[1].flatten() for p in probs])
    y_true_cs = np.concatenate([m[:, :, 1].flatten() for m in masks]) > 0.5

    thresholds = np.arange(0.05, 0.75, 0.05)
    macro_f1_scores = []

    for thr in thresholds:
        f1_gg = f1_score(y_true_gg, (y_prob_gg > thr).astype(int), zero_division=1)
        f1_cs = f1_score(y_true_cs, (y_prob_cs > thr).astype(int), zero_division=1)
        macro_f1_scores.append((f1_gg + f1_cs) / 2)

    best_thr = thresholds[np.argmax(macro_f1_scores)]
    
    # 시각화 그래프 저장
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, macro_f1_scores, marker='o', color='black', label='Macro F1')
    plt.axvline(best_thr, color='blue', linestyle='--', label=f'Best Thr: {best_thr:.2f}')
    plt.title('Threshold Optimization')
    plt.legend(); plt.grid(True); plt.savefig(save_path); plt.close()
    
    return best_thr

def run_validation_analysis(probs, images, masks, config, threshold):
    """상세 지표 리포트 및 시각화 저장"""
    save_dir = os.path.join(config["base_path"], config["run_name"], "val_prediction")
    os.makedirs(save_dir, exist_ok=True)
    
    all_gt_gg, all_pd_gg = [], []
    all_gt_cs, all_pd_cs = [], []

    for i in range(len(images)):
        pred_mask = (probs[i] > threshold).astype(np.uint8)
        # 데이터 수집
        all_gt_gg.append(masks[i][:,:,0].flatten()); all_pd_gg.append(pred_mask[0].flatten())
        all_gt_cs.append(masks[i][:,:,1].flatten()); all_pd_cs.append(pred_mask[1].flatten())
        
        # 샘플 이미지 저장
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        gt_l_gg, gt_l_cn = get_overlay_mask(masks[i], is_ch_first=False)
        pd_l_gg, pd_l_cn = get_overlay_mask(pred_mask, is_ch_first=True)
        axes[0].imshow(images[i].squeeze(), cmap='bone'); axes[0].set_title("CT")
        axes[1].imshow(images[i].squeeze(), cmap='bone'); axes[1].imshow(gt_l_gg); axes[1].imshow(gt_l_cn); axes[1].set_title("GT")
        axes[2].imshow(images[i].squeeze(), cmap='bone'); axes[2].imshow(pd_l_gg); axes[2].imshow(pd_l_cn); axes[2].set_title("Pred")
        for ax in axes: ax.axis('off')
        plt.savefig(f"{save_dir}/val_{i}.png", bbox_inches='tight'); plt.close()

    f1_gg = f1_score(np.concatenate(all_gt_gg)>0.5, np.concatenate(all_pd_gg), zero_division=1)
    f1_cs = f1_score(np.concatenate(all_gt_cs)>0.5, np.concatenate(all_pd_cs), zero_division=1)
    print(f"\n✅ [VAL] Combined F1: {(f1_gg+f1_cs)/2:.4f} (GG: {f1_gg:.4f}, CS: {f1_cs:.4f})")

# ==========================================
# 3. 테스트(Test) 전용 로직
# ==========================================

def run_test_submission(probs, images, config, threshold):
    """테스트 결과 시각화 및 CSV 생성"""
    save_dir = os.path.join(config["base_path"], config["run_name"], "test_prediction")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(images)):
        pred_mask = (probs[i] > threshold).astype(np.uint8)
        l_gg, l_cn = get_overlay_mask(pred_mask, is_ch_first=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(images[i].squeeze(), cmap='bone'); plt.imshow(l_gg); plt.imshow(l_cn)
        plt.title(f"Test Result {i}"); plt.axis('off')
        plt.savefig(f"{save_dir}/test_{i}.png", bbox_inches='tight'); plt.close()

    # Submission 생성
    test_masks = (probs[:, :2, :, :].transpose(0, 2, 3, 1) > threshold).astype(int)
    submission_path = os.path.join(config["base_path"], config["run_name"], "submission.csv")
    pd.DataFrame({'Id': np.arange(test_masks.size), 'Predicted': test_masks.ravel()}).to_csv(submission_path, index=False)
    print(f"✅ [TEST] Submission CSV 생성 완료: {submission_path}")

# ==========================================
# 4. 통합 추론 파이프라인 (The Switch)
# ==========================================

def run_inference_pipeline(config, data_images, data_masks=None, is_valid=True, best_thr=0.5):
    """
    is_valid=True: Threshold 최적화 + 상세 분석 수행
    is_valid=False: 지정된 best_thr를 사용하여 Test 결과 생성
    """
    result_base_dir = os.path.join(config["base_path"], config["run_name"])
    
    # 1. 모델 공통 로드
    print(f"🏗️ 모델 로딩 중 ({'Validation' if is_valid else 'Test'} Mode)...")
    model = MSASkipUnet(config).to(config["device"])
    model_path = os.path.join(result_base_dir, "best.pt")
    checkpoint = torch.load(model_path, map_location=config["device"], weights_only=True)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

    # 2. 추론 수행
    probs = predict_probabilities(model, data_images, config)

    # 3. 모드별 분기 처리
    if is_valid:
        # Validation 모드: 최적 임계값을 찾고 분석
        print("최적 threshold 서칭 중...")
        # thr_curve_path = os.path.join(result_base_dir, "f1_threshold_curve.png")
        # best_thr = find_best_threshold(probs, data_masks, config, thr_curve_path)
        best_thr = 0.55
        print("Validation 시각화 결과 저장 중...")
        run_validation_analysis(probs, data_images, data_masks, config, best_thr)
        return best_thr  # 나중에 Test에서 쓰기 위해 반환
    else:
        # Test 모드: 주어진 임계값으로 결과만 생성
        run_test_submission(probs, data_images, config, best_thr)
        return None

# ==========================================
# 메인 실행
# ==========================================

# 데이터 로드
print("추론을 위해 데이터를 로드합니다")
config = load_config()
X_med = np.load("data/images_medseg.npy")
Y_med = np.load("data/masks_medseg.npy")
X_rad = np.load("data/images_radiopedia.npy")
Y_rad = np.load("data/masks_radiopedia.npy")

X_all = np.concatenate([apply_lung_window(X_med, config), apply_lung_window(X_rad, config)], axis=0)
Y_all = np.concatenate([Y_med, Y_rad], axis=0).astype(np.float32)

X_train, X_val, Y_train, Y_val = train_test_split(X_all, Y_all, test_size=config['validation_size'], random_state=config['seed'], shuffle=True)

# [Step 1] Validation만 실행해서 최적 임계값 얻기
print("Validation data 추론 중...")
found_thr = run_inference_pipeline(config, X_val, Y_val, is_valid=True)

# [Step 2] Test만 실행 (위에서 얻은 임계값 사용)
test_images_medseg = np.load("data/test_images_medseg.npy")
X_test_norm = apply_lung_window(test_images_medseg, config)
if X_test_norm.ndim == 3: X_test_norm = np.expand_dims(X_test_norm, axis=-1)

run_inference_pipeline(config, X_test_norm, is_valid=False, best_thr=found_thr)