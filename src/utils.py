import os
import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_specificity_sensitivity(y_true, y_pred_bin):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return sensitivity, specificity, tn, fp, fn, tp

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)