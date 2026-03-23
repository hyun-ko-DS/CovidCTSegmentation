import os
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
from loss import DiceFocalLoss
from msa import MSASkipUnet
from train import apply_lung_window

single_model = MSASkipUnet().to(config["device"])

# 2. 가중치 로드
SAVE_DIR = os.path.join(BASE_PATH, RUN_NAME)
model_path = os.path.join(SAVE_DIR, "best.pt")

if os.path.exists(model_path):
    print(f"🔄 MSA 모델 가중치 로딩 중... ({model_path})")
    checkpoint = torch.load(model_path, map_location=config["device"])

    # 가중치 딕셔너리 추출
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    single_model.load_state_dict(state_dict)
    single_model.eval()
    print("✅ MSA 모델 로드 완료!")
else:
    print(f"⚠️ 모델 파일을 찾을 수 없습니다: {model_path}")

def find_best_threshold(model, images, masks, save_path='f1_threshold.png'):
    model.eval()
    all_probs = []
    all_gt = []

    print("🔍 검증 데이터 확률값 추출 중 (Inference)...")
    for i in tqdm(range(len(images))):
        img_tensor = torch.from_numpy(images[i]).permute(2, 0, 1).unsqueeze(0).to(CONFIG["device"]).float()
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).cpu().squeeze().numpy()

        all_probs.append(prob[:2]) # (2, 512, 512)
        all_gt.append(masks[i][:,:,:2].transpose(2, 0, 1))

    # 클래스별로 데이터 평탄화 및 준비
    y_prob_gg = np.concatenate([p[0].flatten() for p in all_probs])
    y_true_gg = np.concatenate([g[0].flatten() for g in all_gt]) > 0.5

    y_prob_cs = np.concatenate([p[1].flatten() for p in all_probs])
    y_true_cs = np.concatenate([g[1].flatten() for g in all_gt]) > 0.5

    # 0.05부터 0.7까지 0.05 단위로 탐색
    thresholds = np.arange(0.05, 0.75, 0.05)
    macro_f1_scores = []
    gg_f1_scores = []
    cs_f1_scores = []

    print("📊 Macro F1-score 계산 및 시각화 준비 중...")
    for thr in thresholds:
        f1_gg = f1_score(y_true_gg, (y_prob_gg > thr).astype(int), zero_division=1)
        f1_cs = f1_score(y_true_cs, (y_prob_cs > thr).astype(int), zero_division=1)

        macro_f1 = (f1_gg + f1_cs) / 2

        macro_f1_scores.append(macro_f1)
        gg_f1_scores.append(f1_gg)
        cs_f1_scores.append(f1_cs)

        print(f"Thr {thr:.2f} -> Macro F1: {macro_f1:.4f} (GG: {f1_gg:.4f}, CS: {f1_cs:.4f})")

    # 최적의 임계값 찾기
    best_idx = np.argmax(macro_f1_scores)
    best_thr = thresholds[best_idx]
    best_f1 = macro_f1_scores[best_idx]

    # --- 그래프 시각화 및 저장 ---
    plt.figure(figsize=(12, 7))

    # 클래스별 F1 및 Macro F1 선 그래프
    plt.plot(thresholds, macro_f1_scores, marker='o', linestyle='-', color='black', linewidth=3, label='Macro F1 (Overall)')
    plt.plot(thresholds, gg_f1_scores, marker='s', linestyle='--', color='red', alpha=0.6, label='GG F1')
    plt.plot(thresholds, cs_f1_scores, marker='^', linestyle='--', color='green', alpha=0.6, label='Consol F1')

    # 최적 임계값 강조 (수직선)
    plt.axvline(best_thr, color='blue', linestyle=':', linewidth=2)
    plt.annotate(f'Best Threshold: {best_thr:.2f}\nF1: {best_f1:.4f}',
                 xy=(best_thr, best_f1), xytext=(best_thr+0.05, best_f1-0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, fontweight='bold')

    plt.title('Threshold vs F1-score Optimization', fontsize=16)
    plt.xlabel('Probability Threshold', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # 파일 저장
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    print(f"\n⭐ 최적의 Threshold: {best_thr:.2f} (최대 Macro F1: {best_f1:.4f})")
    print(f"🖼️ 그래프가 {save_path}에 저장되었습니다.")

    return best_thr

def evaluate_and_save_val(model, images, masks, save_dir='/content/val_prediction', threshold=0.5):
    """
    model: 학습된 단일 베스트 모델
    images: X_val (Hold-out 검증 이미지)
    masks: Y_val (Hold-out 검증 마스크)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 지표 계산을 위한 리스트
    all_gt_gg, all_pd_gg = [], []
    all_gt_consol, all_pd_consol = [], []
    total_val_loss = 0

    # 색상 정의 (RGBA)
    RED, GREEN = [1, 0, 0, 0.4], [0, 1, 0, 0.4]

    print(f"🧐 총 {len(images)}장의 검증 데이터 분석 시작...")

    model.eval()
    for i in tqdm(range(len(images))):
        img_np = images[i]
        gt_mask_np = masks[i] # (512, 512, 4)

        # 1. 단일 모델 추론
        input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(CONFIG["device"]).float()

        with torch.no_grad():
            output = model(input_tensor)

            # Loss 계산 (전체 평균 loss 확인용)
            v_loss = criterion(output, torch.from_numpy(gt_mask_np).permute(2,0,1).unsqueeze(0).to(CONFIG["device"]))
            total_val_loss += v_loss.item()

            prob = torch.sigmoid(output).cpu().squeeze().numpy() # (4, 512, 512)

        pred_mask = (prob > threshold).astype(np.uint8)

        # 2. 지표 데이터 수집 (GG: 채널 0, Consol: 채널 1)
        gt_gg = (gt_mask_np[:,:,0] > 0.5).astype(np.uint8).flatten()
        pd_gg = pred_mask[0].flatten()
        gt_consol = (gt_mask_np[:,:,1] > 0.5).astype(np.uint8).flatten()
        pd_consol = pred_mask[1].flatten()

        all_gt_gg.append(gt_gg)
        all_pd_gg.append(pd_gg)
        all_gt_consol.append(gt_consol)
        all_pd_consol.append(pd_consol)

        # 3. 시각화 및 저장
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        def get_overlay(m, is_ch_first=True):
            l_gg, l_cn = np.zeros((512,512,4)), np.zeros((512,512,4))
            if is_ch_first:
                l_gg[m[0]>0], l_cn[m[1]>0] = RED, GREEN
            else:
                l_gg[m[:,:,0]>0.5], l_cn[m[:,:,1]>0.5] = RED, GREEN
            return l_gg, l_cn

        gt_l_gg, gt_l_cn = get_overlay(gt_mask_np, False)
        pd_l_gg, pd_l_cn = get_overlay(pred_mask, True)

        axes[0].imshow(img_np.squeeze(), cmap='bone'); axes[0].set_title("Original CT"); axes[0].axis('off')
        axes[1].imshow(img_np.squeeze(), cmap='bone'); axes[1].imshow(gt_l_gg); axes[1].imshow(gt_l_cn)
        axes[1].set_title("Ground Truth"); axes[1].axis('off')
        axes[2].imshow(img_np.squeeze(), cmap='bone'); axes[2].imshow(pd_l_gg); axes[2].imshow(pd_l_cn)
        axes[2].set_title(f"Prediction (Thr: {threshold})"); axes[2].axis('off')

        plt.savefig(f"{save_dir}/val_sample_{i}.png", bbox_inches='tight')
        plt.close()

    # 4. 최종 메트릭 계산
    def calc_metrics(gt, pd):
        gt_all, pd_all = np.concatenate(gt), np.concatenate(pd)
        f1 = f1_score(gt_all, pd_all, zero_division=1)
        prec = precision_score(gt_all, pd_all, zero_division=1)
        rec = recall_score(gt_all, pd_all, zero_division=1)
        iou = jaccard_score(gt_all, pd_all, zero_division=1)
        return f1, prec, rec, iou

    f1_gg, prec_gg, rec_gg, iou_gg = calc_metrics(all_gt_gg, all_pd_gg)
    f1_cs, prec_cs, rec_cs, iou_cs = calc_metrics(all_gt_consol, all_pd_consol)

    print("\n" + "="*50)
    print("📊 VALIDATION PERFORMANCE REPORT")
    print("="*50)
    print(f"· Avg Val Loss: {total_val_loss/len(images):.4f}")
    print(f"· mIoU (Mean IoU): {(iou_gg + iou_cs)/2:.4f}")
    print("-" * 30)
    print(f"[Ground Glass] F1: {f1_gg:.4f} | Prec: {prec_gg:.4f} | Rec: {rec_gg:.4f} | IoU: {iou_gg:.4f}")
    print(f"[Consolidation] F1: {f1_cs:.4f} | Prec: {prec_cs:.4f} | Rec: {rec_cs:.4f} | IoU: {iou_cs:.4f}")
    print("-" * 30)
    print(f"⭐ FINAL COMBINED F1 (대회 지표): {(f1_gg + f1_cs)/2:.4f}")
    print("="*50)
    print(f"📁 모든 결과 이미지가 {save_dir} 에 저장되었습니다.")

def run_test_inference_single(model, test_images, save_dir='/content/test_prediction', threshold=0.5):
    """
    model: 학습된 단일 베스트 모델 (Hold-out)
    test_images: 정규화가 완료된 테스트 이미지 (N, 512, 512, 1)
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"🧐 {len(test_images)}장의 테스트 이미지 추론 시작 (단일 모델 모드)...")

    # 확률값을 저장할 리스트 (N, 4, 512, 512)
    test_probs = []

    model.eval()
    for i in tqdm(range(len(test_images))):
        img_np = test_images[i] # (512, 512, 1)

        # Tensor 변환 (B, C, H, W)
        input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(CONFIG["device"]).float()

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).cpu().squeeze().numpy() # (4, 512, 512)
            test_probs.append(prob)

        # --- 시각화 및 저장 ---
        RED, GREEN = [1, 0, 0, 0.4], [0, 1, 0, 0.4]
        pred_mask = (prob > threshold).astype(np.uint8)

        l_gg, l_cn = np.zeros((512, 512, 4)), np.zeros((512, 512, 4))
        l_gg[pred_mask[0] > 0] = RED     # Channel 0: Ground Glass
        l_cn[pred_mask[1] > 0] = GREEN   # Channel 1: Consolidation

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np.squeeze(), cmap='bone')
        plt.title(f"Test Image {i}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img_np.squeeze(), cmap='bone')
        plt.imshow(l_gg)
        plt.imshow(l_cn)
        plt.title(f"Prediction (Thr: {threshold})")
        plt.axis('off')

        plt.savefig(f"{save_dir}/test_result_{i}.png", bbox_inches='tight')
        plt.close()

    # --- 2. 대회 포맷 변환 (N, 512, 512, 2) ---
    test_probs_np = np.array(test_probs) # (10, 4, 512, 512)

    # 1. GG와 Consol 클래스(0, 1번 채널)만 선택
    # 2. (N, C, H, W) -> (N, H, W, C)로 Transpose
    test_masks_prediction = test_probs_np[:, :2, :, :].transpose(0, 2, 3, 1)

    # 3. 임계값 적용하여 이진화
    test_masks_prediction = (test_masks_prediction > threshold).astype(int)

    print(f"📊 최종 예측 마스크 형상: {test_masks_prediction.shape}")

    # --- 3. submission.csv 생성 ---
    submission_df = pd.DataFrame(
        data=np.stack((
            np.arange(len(test_masks_prediction.ravel())),
            test_masks_prediction.ravel()
        ), axis=-1),
        columns=['Id', 'Predicted']
    )

    submission_df.set_index('Id').to_csv('submission.csv')
    print("🚀 submission.csv 파일이 성공적으로 생성되었습니다!")

best_thr = find_best_threshold(single_model, X_val, Y_val)
criterion = DiceFocalLoss()
evaluate_and_save_val(single_model, X_val, Y_val, threshold=best_thr)

# 1. 테스트 이미지 로드 (파일 경로를 확인해주세요)
# 보통 np.load('test_images_medseg.npy') 등으로 로드됩니다.
# X_test_raw = np.load(test_images_medseg)

# 2. 학습 데이터와 동일한 윈도잉/정규화 적용
X_test_norm = apply_lung_window(test_images_medseg)

# 3. 모델이 기대하는 차원 (N, 512, 512, 1)로 변경
if len(X_test_norm.shape) == 3: # (10, 512, 512) 인 경우
    X_test_norm = np.expand_dims(X_test_norm, axis=-1)

print(f"✅ 테스트 데이터 준비 완료: {X_test_norm.shape}")
print(f"✅ 데이터 범위: {X_test_norm.min()} ~ {X_test_norm.max()}")

run_test_inference_single(single_model, X_test_norm, threshold=best_thr)