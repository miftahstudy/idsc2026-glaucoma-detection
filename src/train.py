import os
import sys
import gc
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

sys.path.append("src")
from dataset import FundusDataset
from utils import set_seed, ensure_dir

# =========================
# CONFIG
# =========================
SEED = 42
BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-4
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_CSV = os.path.join("splits", "train.csv")
VAL_CSV = os.path.join("splits", "val.csv")
IMG_DIR = os.path.join("data", "Images")
MODEL_DIR = "models"

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best.pth")
LAST_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "last_checkpoint.pth")

ensure_dir(MODEL_DIR)
set_seed(SEED)

print(f"Using device: {DEVICE}")

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_ds = FundusDataset(train_df, IMG_DIR, transform=train_transform)
val_ds = FundusDataset(val_df, IMG_DIR, transform=val_transform)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =========================
# MODEL
# =========================
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# RESUME CHECKPOINT
# =========================
start_epoch = 0
best_val_loss = float("inf")

if os.path.exists(LAST_CHECKPOINT_PATH):
    print("Checkpoint ditemukan. Melanjutkan training dari checkpoint terakhir...")
    checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"Resume dari epoch {start_epoch}")
    print(f"Best val loss sebelumnya: {best_val_loss:.4f}")
else:
    print("Tidak ada checkpoint lama. Training mulai dari awal.")

# =========================
# TRAIN LOOP
# =========================
for epoch in range(start_epoch, EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # ---- TRAIN ----
    model.train()
    train_loss = 0.0

    train_bar = tqdm(train_loader, desc="Training", leave=True)
    for images, labels in train_bar:
        images = images.to(DEVICE)
        labels = labels.unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = train_loss / max(len(train_loader), 1)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Validation", leave=True)
        for images, labels in val_bar:
            images = images.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            val_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_val_loss = val_loss / max(len(val_loader), 1)

    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # ---- SAVE LAST CHECKPOINT (SETIAP EPOCH) ----
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        LAST_CHECKPOINT_PATH
    )
    print(f"Checkpoint terakhir disimpan ke: {LAST_CHECKPOINT_PATH}")

    # ---- SAVE BEST MODEL ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model diupdate: {BEST_MODEL_PATH}")

    # ---- CLEAN MEMORY ----
    del images, labels, outputs, loss
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print("\nTraining selesai.")
print(f"Best model: {BEST_MODEL_PATH}")
print(f"Last checkpoint: {LAST_CHECKPOINT_PATH}")