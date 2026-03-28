import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

sys.path.append("src")
from utils import ensure_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
TEST_CSV = os.path.join("splits", "test.csv")
IMG_DIR = os.path.join("data", "Images")
MODEL_PATH = os.path.join("models", "best.pth")
OUT_DIR = os.path.join("outputs", "gradcam")

ensure_dir(OUT_DIR)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

target_layer = model.layer4[-1]

gradients = []
activations = []

def forward_hook(module, input_tensor, output_tensor):
    activations.append(output_tensor)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

df = pd.read_csv(TEST_CSV)
sample_df = df.sample(min(8, len(df)), random_state=42)

for _, row in sample_df.iterrows():
    img_name = row["Image Name"]
    true_label = row["Label"]

    img_path = os.path.join(IMG_DIR, img_name)
    pil_img = Image.open(img_path).convert("RGB")
    pil_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(pil_resized)

    x = transform(pil_img).unsqueeze(0).to(DEVICE)

    gradients.clear()
    activations.clear()

    out = model(x)
    prob = torch.sigmoid(out)[0, 0].item()

    model.zero_grad()
    out.backward()

    grad = gradients[0].detach().cpu().numpy()[0]
    act = activations[0].detach().cpu().numpy()[0]

    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)

    pred_label = "GON+" if prob >= 0.5 else "GON-"

    save_name = f"{img_name.replace('.jpg','')}_true-{true_label}_pred-{pred_label}_p-{prob:.3f}.jpg"
    save_path = os.path.join(OUT_DIR, save_name)
    cv2.imwrite(save_path, overlay)

print(f"Grad-CAM selesai. File disimpan di: {OUT_DIR}")