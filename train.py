import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import json
from pathlib import Path

import albumentations as A
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import torch
from torchmetrics.functional.classification import multiclass_f1_score
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

class CovidDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx] # (512, 512, 1)
        mask = self.masks[idx]   # (512, 512, 4)

        if self.transform:
            # Albumentations는 HWC 형식을 받으므로 그대로 전달
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # PyTorch 텐서 변환 및 Channel First (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return image, mask

class MSASkipUnet(nn.Module):
    def __init__(self):
        super().__init__()
        # 전역 함수인 get_model_tools()를 호출해 base_model을 가져옵니다.
        base_obj, _, _ = get_model_tools()
        self.base_model = base_obj

        encoder_channels = self.base_model.encoder.out_channels

        self.msa_blocks = nn.ModuleList([
            MSABlock(ch) for ch in encoder_channels
        ])

    def forward(self, x):
        # 인코더는 base_model의 것을 사용
        features = self.base_model.encoder(x)

        # Skip Connection마다 MSA 적용
        msa_features = []
        for i, feat in enumerate(features):
            msa_feat = self.msa_blocks[i](feat)
            msa_features.append(msa_feat)

        # 디코더에 리스트로 전달
        decoder_output = self.base_model.decoder(msa_features)
        masks = self.base_model.segmentation_head(decoder_output)

        return masks

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def load_config(config_path="config.json"):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def apply_lung_window(images):
    # 1. 타입 변환
    images = images.astype(np.float32)

    # 2. 폐 영역 윈도잉
    window_min = config['window_min']
    window_max = config['window_max']

    # 범위를 벗어나는 값은 잘라냄 (Clipping)
    images = np.clip(images, window_min, window_max)

    # 3. 0~1 범위로 정규화 (전체 데이터 동일 기준)
    images = (images - window_min) / (window_max - window_min)
    return images

# 모델 생성 함수
def get_model_tools():
    # SMP 라이브러리를 이용한 U-Net 선언
    model = smp.Unet(
        encoder_name=config["encoder_name"],
        encoder_weights=config["encoder_weights"],
        in_channels=config["in_channels"],
        classes=config["classes"],
        encoder_depth=config["encoder_depth"],                
        activation=None,
        decoder_attention_type = None
    ).to(config["device"])

    criterion = DiceFocalLoss()

    optimizer = optim.AdamW(model.parameters(), lr=config["LR"], weight_decay = config['weight_decay'])
    return model, criterion, optimizer

def get_msa_unet_tools():
    # 1. 위에서 정의한 MSASkipUnet 클래스로 모델 생성
    model = MSASkipUnet().to(config["device"])
    _, criterion, _ = get_model_tools()

    # 3. 옵티마이저 설정 (AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["LR"]),
        weight_decay=config['weight_decay']
    )
    return model, criterion, optimizer

def train_msa_model_holdout(train_loader, val_loader):
    # wandb 초기화 생략 (필요 시 주석 해제)
    run = wandb.init(
        project="covid-19-segmentation",
        name=f"{RUN_NAME}",
        config=CONFIG,
        reinit=True
    )

    model, criterion, optimizer = get_msa_unet_tools()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    best_model_path = os.path.join(SAVE_DIR, "best.pt")
    last_model_path = os.path.join(SAVE_DIR, "last.pt") # last.pt 경로 추가

    early_stopping = EarlyStopping(patience=config["patience"], verbose=True, path=best_model_path)

    for epoch in range(config["epochs"]):
        current_epoch = epoch + 1
        is_f1_step = (current_epoch % 5 == 0)

        # [Train Step]
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {current_epoch} MSA-Train", leave=False)

        for imgs, msks in train_pbar:
            imgs, msks = imgs.to(config["device"]), msks.to(config["device"])
            if msks.shape[1] != config["classes"]:
                msks = msks.permute(0, 3, 1, 2).float()

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, msks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # [Validation Step]
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(config["device"]), msks.to(config["device"])
                if msks.shape[1] != config["classes"]:
                    msks = msks.permute(0, 3, 1, 2).float()

                outputs = model(imgs)
                v_loss = criterion(outputs, msks)
                val_loss += v_loss.item()

                if is_f1_step:
                    preds = torch.argmax(outputs, dim=1)
                    targets = torch.argmax(msks, dim=1)
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.cpu())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # --- [추가] last.pt 저장 로직 ---
        # 매 에폭마다 모델의 현재 상태를 저장합니다.
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }, last_model_path)

        metrics = {
            "epoch": current_epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]['lr']
        }

        if is_f1_step:
            y_pred = torch.cat(all_preds).view(-1)
            y_true = torch.cat(all_targets).view(-1)
            f1_scores = multiclass_f1_score(y_pred, y_true, num_classes=config["classes"], average=None)

            class_names = ["GGO", "Consol", "C3", "C4"]
            f1_dict = {f"f1_{class_names[i]}": f1_scores[i].item() for i in range(len(f1_scores))}
            f1_target_mean = (f1_scores[0] + f1_scores[1]).item() / 2
            f1_dict["f1_target_mean"] = f1_target_mean

            metrics.update(f1_dict)
            print(f"\n✨ Epoch {current_epoch} Metrics:")
            print(f"📊 [GGO: {f1_dict['f1_GGO']:.4f}] [Consol: {f1_dict['f1_Consol']:.4f}] [Target Mean: {f1_target_mean:.4f}]")

        wandb.log(metrics)
        scheduler.step()

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping triggered at epoch {current_epoch}")
            break
    run.finish()
    return model

def run_training_pipeline(
    path, config_path="config.json", base_path="./results", run_name="exp_25_best_model"):

    # 0. 설정 로드 및 실행 컨텍스트 설정
    global config
    config = load_config(config_path)
    RUN_NAME = run_name
    SAVE_DIR = os.path.join(base_path, run_name)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 이미지 로드 및 전처리
    images_medseg = np.load(os.path.join("data", "images_medseg.npy"))
    masks_medseg = np.load(os.path.join("data", "masks_medseg.npy"))
    images_radiopedia = np.load(os.path.join("data", "images_radiopedia.npy"))
    masks_radiopedia = np.load(os.path.join("data", "masks_radiopedia.npy"))
    test_images_medseg = np.load(os.path.join("data", "test_images_medseg.npy"))

    print("모든 데이터 로드 완료!\n")
    # print_stats("Images Medseg", images_medseg)
    # print_stats("Masks Medseg", masks_medseg)
    # print_stats("Images Radiopedia", images_radiopedia)
    # print_stats("Masks Radiopedia", masks_radiopedia)
    # print_stats("Test Images", test_images_medseg)

    X_med_norm = apply_lung_window(images_medseg)
    X_rad_norm = apply_lung_window(images_radiopedia)
    X_all = np.concatenate([X_med_norm, X_rad_norm], axis=0)
    Y_all = np.concatenate([masks_medseg, masks_radiopedia], axis=0).astype(np.float32)

    # 2. 데이터 증강
    train_transform = A.Compose([
        A.HorizontalFlip(p=config["horizontal_flip_p"]),
        A.VerticalFlip(p=config["vertical_flip_p"]),
        A.ShiftScaleRotate(
            shift_limit=config["shift_limit"],
            scale_limit=config["scale_limit"],
            rotate_limit=config["rotate_limit"],
            p=config["rotate_p"],
        ),
        A.ElasticTransform(
            alpha=config["elastic_transform_alpha"],
            sigma=config["elastic_transform_sigma"],
            alpha_affine=config["elastic_transform_alpha_affine"],
            p=config["elastic_transform_p"],
        ),
        A.RandomBrightnessContrast(
            brightness_limit=config["random_brightness_limit"],
            contrast_limit=config["random_contrast_limit"],
            p=config["random_brightness_contrast_p"],
        ),
    ])

    X_train, X_val, Y_train, Y_val = train_test_split(
    X_all, Y_all,
    test_size=config['validation_size'],
    random_state=config['seed'],
    shuffle=True)


    # 3. 데이터셋, 데이터로더 정의
    train_dataset = CovidDataset(X_train, Y_train, transform=train_transform)
    val_dataset = CovidDataset(X_val, Y_val, transform=None)
    print("✅ 데이터 분할 완료")
    print(f"🏠 Train: {len(train_dataset)}장 | 🏥 Val: {len(val_dataset)}장")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    print("✅ 데이터로더 정의 완료")

    # 4. 모델 학습
    # best_trained_model = train_msa_model_holdout(train_loader, val_loader)
    # return best_trained_model

