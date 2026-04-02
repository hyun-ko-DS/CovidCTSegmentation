import os
import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
from loss import DiceFocalLoss
from msa import MSABlock, MSASkipUnet
from train import apply_lung_window

def find_best_threshold(model, images, masks, config, save_path):
    model.eval()
    all_probs = []
    all_gt = []

    print("🔍 검증 데이터 확률값 추출 중 (Inference)...")
    for i in tqdm(range(len(images))):
        img_tensor = torch.from_numpy(images[i]).permute(2, 0, 1).unsqueeze(0).to(config["device"]).float()
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).cpu().squeeze().numpy()

        all_probs.append(prob[:2]) 
        all_gt.append(masks[i][:,:,:2].transpose(2, 0, 1))

    y_prob_gg = np.concatenate([p[0].flatten() for p in all_probs])
    y_true_gg = np.concatenate([g[0].flatten() for g in all_gt]) > 0.5
    y_prob_cs = np.concatenate([p[1].flatten() for p in all_probs])
    y_true_cs = np.concatenate([g[1].flatten() for g in all_gt]) > 0.5

    thresholds = np.arange(0.05, 0.75, 0.05)
    macro_f1_scores, gg_f1_scores, cs_f1_scores = [], [], []

    print("📊 Macro F1-score 계산 및 시각화 준비 중...")
    for thr in thresholds:
        f1_gg = f1_score(y_true_gg, (y_prob_gg > thr).astype(int), zero_division=1)
        f1_cs = f1_score(y_true_cs, (y_prob_cs > thr).astype(int), zero_division=1)
        macro_f1 = (f1_gg + f1_cs) / 2
        macro_f1_scores.append(macro_f1)
        gg_f1_scores.append(f1_gg)
        cs_f1_scores.append(f1_cs)

    best_idx = np.argmax(macro_f1_scores)
    best_thr = thresholds[best_idx]
    best_f1 = macro_f1_scores[best_idx]

    plt.figure(figsize=(12, 7))
    plt.plot(thresholds, macro_f1_scores, marker='o', color='black', linewidth=3, label='Macro F1')
    plt.plot(thresholds, gg_f1_scores, marker='s', color='red', alpha=0.6, label='GG F1')
    plt.plot(thresholds, cs_f1_scores, marker='^', color='green', alpha=0.6, label='Consol F1')
    plt.axvline(best_thr, color='blue', linestyle=':')
    plt.title('Threshold vs F1-score Optimization')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"\n⭐ 최적의 Threshold: {best_thr:.2f} (최대 Macro F1: {best_f1:.4f})")
    return best_thr

def evaluate_and_save_val(model, images, masks, config, save_dir, threshold=0.5):
    """
    model: 학습된 단일 베스트 모델
    images: X_val (검증 이미지)
    masks: Y_val (검증 마스크)
    config: 설정 딕셔너리
    save_dir: 저장 경로 (런팟 경로 유지)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 지표 계산을 위한 리스트
    all_gt_gg, all_pd_gg = [], []
    all_gt_cs, all_pd_cs = [], []
    total_val_loss = 0
    
    # 장치 및 손실함수 설정
    device = config["device"]
    criterion = DiceFocalLoss(config)

    # 색상 정의 (RGBA)
    RED, GREEN = [1, 0, 0, 0.4], [0, 1, 0, 0.4]

    print(f"🧐 총 {len(images)}장의 검증 데이터 분석 및 시각화 시작...")

    model.eval()
    for i in tqdm(range(len(images))):
        img_np = images[i]
        gt_mask_np = masks[i] # (512, 512, 4)

        # 1. 모델 추론
        input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device).float()

        with torch.no_grad():
            output = model(input_tensor)
            # Loss 계산
            v_loss = criterion(output, torch.from_numpy(gt_mask_np).permute(2,0,1).unsqueeze(0).to(device))
            total_val_loss += v_loss.item()
            prob = torch.sigmoid(output).cpu().squeeze().numpy() # (4, 512, 512)

        pred_mask = (prob > threshold).astype(np.uint8)

        # 2. 지표 데이터 수집 (GG: 0, Consol: 1)
        gt_gg = (gt_mask_np[:,:,0] > 0.5).astype(np.uint8).flatten()
        pd_gg = pred_mask[0].flatten()
        gt_cs = (gt_mask_np[:,:,1] > 0.5).astype(np.uint8).flatten()
        pd_cs = pred_mask[1].flatten()

        all_gt_gg.append(gt_gg); all_pd_gg.append(pd_gg)
        all_gt_cs.append(gt_cs); all_pd_cs.append(pd_cs)

        # 3. 시각화 및 저장 (코랩 형식 복구)
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

    # 4. 최종 리포트 출력
    def calc_metrics(gt, pd):
        gt_all, pd_all = np.concatenate(gt), np.concatenate(pd)
        f1 = f1_score(gt_all, pd_all, zero_division=1)
        iou = jaccard_score(gt_all, pd_all, zero_division=1)
        return f1, iou

    f1_gg, iou_gg = calc_metrics(all_gt_gg, all_pd_gg)
    f1_cs, iou_cs = calc_metrics(all_gt_cs, all_pd_cs)

    print("\n" + "="*50)
    print("📊 VALIDATION PERFORMANCE REPORT")
    print("="*50)
    print(f"· Avg Val Loss: {total_val_loss/len(images):.4f}")
    print(f"· mIoU: {(iou_gg + iou_cs)/2:.4f}")
    print(f"[Ground Glass] F1: {f1_gg:.4f} | IoU: {iou_gg:.4f}")
    print(f"[Consolidation] F1: {f1_cs:.4f} | IoU: {iou_cs:.4f}")
    print(f"⭐ FINAL COMBINED F1: {(f1_gg + f1_cs)/2:.4f}")
    print("="*50)

def run_test_inference_single(model, test_images, config, save_dir, threshold=0.5):
    """
    테스트 데이터 시각화도 코랩 오버레이 형식으로 복구
    """
    os.makedirs(save_dir, exist_ok=True)
    test_probs = []
    RED, GREEN = [1, 0, 0, 0.4], [0, 1, 0, 0.4]

    print(f"🧐 {len(test_images)}장의 테스트 이미지 추론 및 시각화 시작...")
    model.eval()
    for i in tqdm(range(len(test_images))):
        img_np = test_images[i]
        input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(config["device"]).float()
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).cpu().squeeze().numpy()
            test_probs.append(prob)

        pred_mask = (prob > threshold).astype(np.uint8)
        
        # 오버레이 생성
        l_gg = np.zeros((512, 512, 4)); l_cn = np.zeros((512, 512, 4))
        l_gg[pred_mask[0] > 0] = RED
        l_cn[pred_mask[1] > 0] = GREEN

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1); plt.imshow(img_np.squeeze(), cmap='bone'); plt.title(f"Test Image {i}"); plt.axis('off')
        plt.subplot(1, 2, 2); plt.imshow(img_np.squeeze(), cmap='bone'); plt.imshow(l_gg); plt.imshow(l_cn)
        plt.title(f"Prediction (Thr: {threshold})"); plt.axis('off')

        plt.savefig(f"{save_dir}/test_result_{i}.png", bbox_inches='tight')
        plt.close()

    # Submission 생성 로직 유지
    test_masks_prediction = (np.array(test_probs)[:, :2, :, :].transpose(0, 2, 3, 1) > threshold).astype(int)
    submission_path = os.path.join(config["base_path"], config["run_name"], "submission.csv")
    pd.DataFrame({'Id': np.arange(len(test_masks_prediction.ravel())), 'Predicted': test_masks_prediction.ravel()}).to_csv(submission_path, index=False)
    print(f"🚀 submission.csv 생성 완료: {submission_path}")

def run_inference_pipeline(config, X_val, Y_val, test_images_medseg):
    # 결과가 저장될 절대 경로 설정
    result_base_dir = os.path.join(config["base_path"], config["run_name"])
    
    # 1. 모델 로드
    model = MSASkipUnet(config).to(config["device"])
    model_path = os.path.join(result_base_dir, "best.pt")
    checkpoint = torch.load(model_path, map_location=config["device"], weights_only=True)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()

    # 2. Step 1: Threshold 최적화
    best_thr_path = os.path.join(result_base_dir, "f1_threshold.png")
    best_thr = find_best_threshold(model, X_val, Y_val, config, best_thr_path)

    # 3. Step 2: Validation 상세 평가 및 시각화
    val_save_dir = os.path.join(result_base_dir, "val_prediction")
    evaluate_and_save_val(model, X_val, Y_val, config, val_save_dir, threshold=best_thr)

    # 4. Step 3: Test 추론 및 submission 생성
    test_save_dir = os.path.join(result_base_dir, "test_prediction")
    X_test_norm = apply_lung_window(test_images_medseg, config)
    if X_test_norm.ndim == 3: X_test_norm = np.expand_dims(X_test_norm, axis=-1)
    run_test_inference_single(model, X_test_norm, config, test_save_dir, threshold=best_thr)

# 4. 추론 파이프라인 실행
run_inference_pipeline(
    config=config, 
    X_val=X_val, 
    Y_val=Y_val, 
    test_images_medseg=test_images_medseg
)