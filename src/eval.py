import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt


sys.path.append("src")
from dataset import FundusDataset
from utils import calculate_specificity_sensitivity, ensure_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 4

TEST_CSV = os.path.join("splits", "test.csv")
IMG_DIR = os.path.join("data", "Images")
MODEL_PATH = os.path.join("models", "best.pth")
OUT_DIR = "outputs"

ensure_dir(OUT_DIR)

test_df = pd.read_csv(TEST_CSV)

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

test_ds = FundusDataset(test_df, IMG_DIR, transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# PENTING: samakan dengan train.py -> ResNet18
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

y_true = []
y_prob = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().tolist())

y_true = np.array(y_true, dtype=int)
y_prob = np.array(y_prob)
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_true, y_prob)
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
sensitivity, specificity, tn, fp, fn, tp = calculate_specificity_sensitivity(y_true, y_pred)

print(f"AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

metrics = {
    "auc": float(auc),
    "accuracy": float(acc),
    "f1_score": float(f1),
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
}

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["GON-", "GON+"])
plt.yticks([0, 1], ["GON-", "GON+"])
plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()

fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=200)
plt.close()

print("Hasil evaluasi disimpan di folder outputs/")