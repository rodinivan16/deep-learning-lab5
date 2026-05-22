import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
!pip install -q pytorch_metric_learning timm
import timm
from pytorch_metric_learning import losses
import kagglehub

from google.colab import drive

# ============================================================
# SETTINGS
# ============================================================
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Настройки обучения
MODEL_NAME = "convnext_tiny"
IMAGE_SIZE = 280
EMB_SIZE = 512
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================
# KAGGLE DOWNLOAD
# ============================================================
print("\nDownloading dataset...")
os.environ["KAGGLE_USERNAME"] = "ivanr0din"
os.environ["KAGGLE_KEY"] = "a4316f6e1cd454fa3c8fddf45d14f245"

dataset_path = kagglehub.competition_download("dl-lab-5-metric-learning")
BASE_DIR = Path(dataset_path)

TRAIN_ROOT = BASE_DIR / "train" / "train"
TEST_ROOT = BASE_DIR / "test_kaggle" / "test_kaggle"
INPUT_SUBMISSION_PATH = BASE_DIR / "submission.csv"

# ----- GOOGLE ДИСК -----
drive.mount('/content/drive')
SAVE_DIR = Path('/content/drive/MyDrive/laba5')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Изменил названия файлов, чтобы отследить эксперимент с GeM
OUTPUT_SUBMISSION_PATH = SAVE_DIR / "submission_convnext_gem.csv"
WEIGHTS_PATH = SAVE_DIR / "model_convnext_gem.pth"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ============================================================
# DATASET & DATALOADERS (100% DATA)
# ============================================================
class ProductDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]

print("\nPreparing dataset (100% for training)...")
all_classes = sorted([p.name for p in TRAIN_ROOT.iterdir() if p.is_dir()])
class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
num_classes = len(all_classes)

train_paths, train_labels = [], []

for cls_name in all_classes:
    cls_idx = class_to_idx[cls_name]
    imgs = [p for p in (TRAIN_ROOT / cls_name).glob("*.*") if p.suffix.lower() in IMAGE_EXTS]

    train_paths.extend(imgs)
    train_labels.extend([cls_idx] * len(imgs))

print(f"Total training images: {len(train_paths)}")
print(f"Total classes (products): {num_classes}")

train_transform = T.Compose([
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(ProductDataset(train_paths, train_labels, train_transform),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# MODEL DEFINITION (С ДОБАВЛЕНИЕМ GeM POOLING)
# ============================================================

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)


class MetricModel(nn.Module):
    def __init__(self, model_name, emb_size):
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')

        self.pooling = GeM()

        dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        backbone_out_channels = self.backbone(dummy).shape[1]

        self.neck = nn.Sequential(
            nn.Linear(backbone_out_channels, emb_size),
            nn.BatchNorm1d(emb_size)
        )

    def forward(self, x):
        features = self.backbone(x)                 # Выход: [B, C, H, W]
        pooled = self.pooling(features)             # Выход: [B, C, 1, 1]
        pooled_flat = pooled.flatten(1)             # Выход: [B, C] 
        embeddings = self.neck(pooled_flat)
        return embeddings

print(f"\nInitializing model {MODEL_NAME} with GeM Pooling...")
model = MetricModel(MODEL_NAME, EMB_SIZE).to(DEVICE)

loss_fn = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=EMB_SIZE, margin=26.7, scale=64).to(DEVICE)

optimizer = torch.optim.AdamW([
    {'params': model.parameters()},
    {'params': loss_fn.parameters(), 'lr': LR * 10}
], lr=LR, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ============================================================
# FULL TRAINING LOOP
# ============================================================
print("\nStarting Full Training...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        embeddings = model(images)
        loss = loss_fn(embeddings, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    scheduler.step()
    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1} finished! Average Train Loss = {train_loss:.4f}")

torch.save(model.state_dict(), WEIGHTS_PATH)
print(f"--> Saved FINAL model to {WEIGHTS_PATH}!")

# ============================================================
# INFERENCE & SUBMISSION
# ============================================================
print("\nStarting Inference on Test set...")
model.eval()

submission = pd.read_csv(INPUT_SUBMISSION_PATH)[["id", "file_1", "file_2"]].copy()

def find_images(root: Path):
    filename_to_path = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            filename_to_path[path.name] = path
    return filename_to_path

filename_to_path = find_images(TEST_ROOT)
needed_files = set(submission["file_1"].astype(str)) | set(submission["file_2"].astype(str))
needed_paths = [filename_to_path[x] for x in sorted(needed_files)]

print(f"\nExtracting embeddings for {len(needed_paths)} test images (Strict 1 Pass)...")
test_embeddings = {}

with torch.no_grad():
    for start in tqdm(range(0, len(needed_paths), BATCH_SIZE)):
        batch_paths = needed_paths[start : start + BATCH_SIZE]

        images_pil = [Image.open(p).convert("RGB") for p in batch_paths]
        images_tensor = torch.stack([test_transform(img) for img in images_pil]).to(DEVICE)

        embs = model(images_tensor)
        embs = F.normalize(embs, p=2, dim=1).cpu().numpy()

        for i, path in enumerate(batch_paths):
            test_embeddings[path.name] = embs[i].astype(np.float32)

print("\nCalculating similarities...")
similarities = []
for row in tqdm(submission.itertuples(index=False), total=len(submission)):
    emb1 = test_embeddings[row.file_1]
    emb2 = test_embeddings[row.file_2]
    sim = float(np.dot(emb1, emb2))
    similarities.append(sim)


sims_array = np.array(similarities)
submission["similarity"] = (sims_array + 1.0) / 2.0

submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
print(f"\nDONE! Saved to: {OUTPUT_SUBMISSION_PATH}")
