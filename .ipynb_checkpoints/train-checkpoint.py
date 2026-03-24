from loss import DiceFocalLoss
from msa import MSABlock, MSASkipUnet

import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pathlib import Path

import albumentations as A
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import torch
from torchmetrics.functional.classification import multiclass_f1_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import wandb

# 1. 데이터셋 정의
class CovidDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        return image, mask

# 2. MSA 모델 정의 (base_model 생성 로직 최적화)
class MSASkipUnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 내부에서 직접 생성하여 중복 호출 및 불필요한 객체 생성 방지
        self.base_model = smp.Unet(
            encoder_name=config["encoder_name"],
            encoder_weights=config["encoder_weights"],
            in_channels=config["in_channels"],
            classes=config["classes"],
            encoder_depth=config["encoder_depth"],                 
            activation=None
        )
        
        encoder_channels = self.base_model.encoder.out_channels
        self.msa_blocks = nn.ModuleList([
            MSABlock(ch) for ch in encoder_channels
        ])

    def forward(self, x):
        features = self.base_model.encoder(x)
        msa_features = []
        for i, feat in enumerate(features):
            msa_feat = self.msa_blocks[i](feat)
            msa_features.append(msa_feat)

        decoder_output = self.base_model.decoder(msa_features)
        masks = self.base_model.segmentation_head(decoder_output)
        return masks

# 3. 조기 종료 클래스
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

# 4. 유틸리티 함수
def load_config(config_path="config.json"):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def apply_lung_window(images, config):
    images = images.astype(np.float32)
    window_min, window_max = config['window_min'], config['window_max']
    images = np.clip(images, window_min, window_max)
    return (images - window_min) / (window_max - window_min)

# 5. 모델 및 도구 생성 (핵심 수정 구간)
def get_msa_unet_tools(config):
    # 모델 생성 시 config 주입
    model = MSASkipUnet(config).to(config["device"])
    
    # Loss 함수에 config 주입 (TypeError 해결)
    criterion = DiceFocalLoss(config) 

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config["LR"]), 
        weight_decay=config['weight_decay']
    )
    return model, criterion, optimizer

# 6. 학습 루프
def train_model(train_loader, val_loader, config, run_name, save_dir):
    run = wandb.init(
        project="covid-19-segmentation",
        name=run_name,
        config=config,
        reinit=True
    )

    model, criterion, optimizer = get_msa_unet_tools(config)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["EPOCHS"])
    
    best_model_path = os.path.join(save_dir, "best.pt")
    last_model_path = os.path.join(save_dir, "last.pt")
    early_stopping = EarlyStopping(patience=config["PATIENCE"], verbose=True, path=best_model_path)

    for epoch in range(config["EPOCHS"]):
        current_epoch = epoch + 1
        is_f1_step = (current_epoch % 5 == 0)

        # [Train]
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

        # [Validation]
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
                    all_preds.append(torch.argmax(outputs, dim=1).cpu())
                    all_targets.append(torch.argmax(msks, dim=1).cpu())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 모델 저장
        torch.save({'epoch': current_epoch, 'model_state_dict': model.state_dict()}, last_model_path)

        metrics = {"epoch": current_epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "lr": optimizer.param_groups[0]['lr']}

        if is_f1_step:
            y_pred, y_true = torch.cat(all_preds).view(-1), torch.cat(all_targets).view(-1)
            f1_scores = multiclass_f1_score(y_pred, y_true, num_classes=config["classes"], average=None)
            
            class_names = ["GGO", "Consol", "C3", "C4"]
            for i, score in enumerate(f1_scores):
                metrics[f"f1_{class_names[i]}"] = score.item()
            metrics["f1_target_mean"] = (f1_scores[0] + f1_scores[1]).item() / 2
            
            print(f"\n✨ Epoch {current_epoch} F1 GGO: {metrics['f1_GGO']:.4f}, Consol: {metrics['f1_Consol']:.4f}")

        wandb.log(metrics)
        scheduler.step()
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop: break

    run.finish()
    return model

# 7. 메인 파이프라인
def run_training_pipeline(path = None,config_path="config.json", base_path="./results", run_name="exp_final"):
    config = load_config(config_path)
    save_dir = os.path.join(base_path, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # 데이터 로드 (경로는 사용자 환경에 맞춰 수정 필요)
    X_med = np.load("data/images_medseg.npy")
    Y_med = np.load("data/masks_medseg.npy")
    X_rad = np.load("data/images_radiopedia.npy")
    Y_rad = np.load("data/masks_radiopedia.npy")

    X_all = np.concatenate([apply_lung_window(X_med, config), apply_lung_window(X_rad, config)], axis=0)
    Y_all = np.concatenate([Y_med, Y_rad], axis=0).astype(np.float32)

    X_train, X_val, Y_train, Y_val = train_test_split(X_all, Y_all, test_size=config['validation_size'], random_state=config['seed'], shuffle=True)

    train_transform = A.Compose([
        A.HorizontalFlip(p=config["horizontal_flip_p"]),
        A.VerticalFlip(p=config["vertical_flip_p"]),
        A.ShiftScaleRotate(shift_limit=config["shift_limit"], scale_limit=config["scale_limit"], rotate_limit=config["rotate_limit"], p=config["rotate_p"]),
        A.RandomBrightnessContrast(p=config["random_brightness_contrast_p"])
    ])

    train_loader = DataLoader(CovidDataset(X_train, Y_train, train_transform), batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(CovidDataset(X_val, Y_val), batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["num_workers"])

    return train_model(train_loader, val_loader, config, run_name, save_dir)