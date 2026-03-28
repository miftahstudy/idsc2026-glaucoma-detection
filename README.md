# 👁️ Glaucoma Screening & Triage Support System using HYGD

Deep learning-based screening and triage-support system for detecting glaucomatous optic neuropathy (GON) from retinal fundus images using the Hillel Yaffe Glaucoma Dataset (HYGD).

Developed for **International Data Science Challenge (IDSC) 2026**  
Theme: **Mathematics for Hope in Healthcare**

---

## 🧠 Project Overview

This project presents a deep learning-based system for detecting glaucomatous optic neuropathy (GON) from retinal fundus images using the HYGD dataset.

Unlike traditional approaches that focus solely on classification performance, this work is designed as a **clinically meaningful screening and triage-support system**, enabling:

- risk-based patient prioritization
- early detection of glaucoma
- interpretable predictions via Grad-CAM
- uncertainty-aware decision support

---

## 🌍 Clinical Motivation

Glaucoma is a leading cause of irreversible blindness worldwide and is often asymptomatic in early stages. Approximately 50% of cases remain undiagnosed until significant vision loss occurs.

In many regions, especially resource-limited settings:

- access to ophthalmologists is limited
- screening is delayed or unavailable
- patient prioritization is not optimized

This project aims to bridge that gap by providing a **deployable AI-based screening tool** that supports early identification and efficient triage.

---

## 📊 Dataset

- **Dataset:** Hillel Yaffe Glaucoma Dataset (HYGD) v1.0.0
- **Source:** PhysioNet
- **Total Images:** 747
- **Patients:** 288
- **Labels:** GON+ / GON-
- **Metadata:** Patient ID, Quality Score

> Labels are based on comprehensive ophthalmic examinations (OCT, visual field tests, and follow-up), ensuring high clinical reliability.

---

## 🏗️ System Design

The system consists of:

- Patient-level data splitting (anti data leakage)
- Image preprocessing and normalization
- Deep learning model for binary classification
- Probability-based risk scoring
- Grad-CAM interpretability
- Threshold-based triage decision support

This design transforms a classification model into a **clinically usable decision support system**.

---

## 🔒 Data Splitting Strategy (Critical)

We apply a strict **patient-level split** to prevent data leakage.

Since multiple images may belong to the same patient, image-level splitting would artificially inflate performance. Therefore, all images from a patient are assigned to a single subset (train/validation/test), ensuring realistic evaluation on unseen patients.

---

## 🤖 Model Architecture

- Backbone: **ResNet18 (ImageNet pretrained)**
- Output: Single neuron (binary classification)
- Activation: Sigmoid (probability output)

ResNet18 is chosen for its balance between:

- performance
- stability
- computational efficiency

This makes it suitable for **deployment in limited-resource environments**.

---

## ⚙️ Training Strategy

- Loss: Binary Cross Entropy with Logits
- Optimizer: Adam
- Checkpoint saving for fault-tolerant training
- Hardware-aware configuration

---

## 📈 Performance

Evaluated on a **held-out patient-level test set**:

- AUC: **0.9940**
- Accuracy: **0.9434**
- Sensitivity: **0.9615**
- Specificity: **0.8929**
- F1-score: **0.9615**

These results demonstrate strong discriminative ability while maintaining **high sensitivity**, which is critical for screening safety.

---

## 🔍 Interpretability (Grad-CAM)

Grad-CAM is used to visualize model attention:

- GON+ → focused activation near optic disc
- GON- → diffuse or minimal activation

Supports:

- transparency
- clinical trust
- validation of learned features

---

## ⚠️ Uncertainty & Decision Threshold

- Default threshold: **0.5**
- Adjustable threshold enables:
  - higher sensitivity (screening safety)
  - balanced specificity

Predictions in the **0.4-0.6 range** are considered borderline and should be manually reviewed.

---

## 🏥 Clinical Workflow Integration

1. Fundus image is captured
2. Model predicts GON probability
3. Patients are categorized:
   - High-risk → urgent referral
   - Borderline → further review
   - Low-risk → routine monitoring

---

## 🌟 Real-World Impact

- Early glaucoma detection
- Risk-based prioritization
- Expanded access in underserved regions
- Efficient use of specialist resources

---

## ⚖️ Design Philosophy

> **Practical, interpretable, and deployable**

Prioritizes:

- robustness
- simplicity
- real-world reliability

---

# 🚀 Extended Pipeline & Application

## Overview

This repository contains an end-to-end pipeline for glaucoma screening:

- patient-level splitting
- probabilistic prediction
- Grad-CAM visualization
- risk stratification
- image-quality warnings
- Streamlit app with EDA dashboard

---

## Main Features

- **Patient-level split**
- **ResNet18 classifier**
- **Held-out test evaluation**
- **Grad-CAM visualization**
- **Risk categories (4-level)**
- **Quality-aware warnings**
- **Interactive Streamlit app**

---

## 📂 Repository Structure

```text
.
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── Images/
│   └── Labels.csv
│
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── models/
│   ├── best.pth
│   └── last_checkpoint.pth
│
├── outputs/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── gradcam/
│
└── src/
    ├── dataset.py
    ├── utils.py
    ├── split.py
    ├── train.py
    ├── eval.py
    └── gradcam.py
```

## 💻 Environment Requirements

### Software

- Python 3.10+
- pip

### Hardware

- CPU supported
- GPU recommended

---

## ⚙️ Installation

```bash
git clone <your-repository-url>
cd <your-repository-folder>

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

## ▶️ How to Run

### 1. Split dataset
```bash
python src/split.py
```

### 2. Train model
```bash
python src/train.py
```

### 3. Evaluate
```bash
python src/eval.py
```

### 4. Grad-CAM
```bash
python src/gradcam.py
```

### 5. Launch app
```bash
streamlit run app.py
```

---

## 📊 Streamlit App Features

### 1. Prediction

- Upload image
- Probability output
- Risk category
- Clinical recommendation
- Grad-CAM visualization
- Quality warnings

### 2. EDA & Insights

- Dataset distribution
- Patient statistics
- Quality analysis
- Sample images

---

## 🧪 Model Details

- Backbone: ResNet18
- Loss: BCEWithLogitsLoss
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 4
- Epochs: 5
- Input size: 224×224

---

## 📊 Evaluation Metrics

- AUC
- Accuracy
- Sensitivity
- Specificity
- F1-score
- ROC Curve
- Confusion Matrix

---

## 🧠 Risk Stratification

- **High Risk:** p ≥ 0.85
- **Moderate:** 0.50-0.85
- **Borderline:** 0.35-0.50
- **Low Risk:** p < 0.35

---

## 🧪 Quality-Aware Warnings

Quality checks include:

- Blur (Laplacian variance)
- Brightness (mean intensity)
- Contrast (standard deviation)

Categories:

- Good
- Fair
- Poor

---

## 🔁 Reproducibility

```bash
python src/split.py
python src/train.py
python src/eval.py
python src/gradcam.py
```

Ensure the HYGD dataset is placed in `data/`.

---

## ⚠️ Limitations

- Small dataset (747 images)
- Single-center bias
- Binary classification only
- No external validation
- Heuristic quality module

---

## 📚 Citation

If you use this project or the dataset, please cite the following:

### Dataset Citation (HYGD)

Abramovich, O., Pizem, H., Fhima, J., Berkowitz, E., Gofrit, B., Van Eijgen, J., Blumenthal, E., & Behar, J. (2025). Hillel Yaffe Glaucoma Dataset (HYGD): A Gold-Standard Annotated Fundus Dataset for Glaucoma Detection (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/z0ak-km33

---

### PhysioNet Platform Citation

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

---

### Additional Reference

Abramovich, Or, et al. (2025) “GONet: A Generalizable Deep Learning Model for Glaucoma Detection.” arXiv.

---

## ⚠️ Disclaimer

This project is developed for research and competition purposes only.  
It is not intended to replace professional medical diagnosis or clinical decision-making.
