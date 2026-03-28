# 👁️ Glaucoma Screening & Triage Support System using HYGD

AI-based screening and triage-support system for detecting glaucomatous optic neuropathy (GON) from retinal fundus images using the Hillel Yaffe Glaucoma Dataset (HYGD).

Developed for **International Data Science Challenge (IDSC) 2026**  
Theme: **Mathematics for Hope in Healthcare**

---

## 🧠 Project Overview

This project presents a deep learning–based system for glaucoma detection that goes beyond standard classification.

Instead of focusing solely on prediction accuracy, this work is designed as a **clinically meaningful screening and triage-support system**, enabling:

- risk-based patient prioritization  
- early detection of glaucoma  
- interpretable predictions via Grad-CAM  
- uncertainty-aware decision support  

---

## 🌍 Clinical Motivation

Glaucoma is a leading cause of irreversible blindness worldwide and is often asymptomatic in early stages.

Approximately **50% of cases remain undiagnosed** until significant vision loss occurs.

In many regions:

- access to ophthalmologists is limited  
- screening is delayed or unavailable  
- patient prioritization is inefficient  

This project aims to address these challenges through a **deployable AI-based screening tool**.

---

## ⚡ Quick Start

```bash
git clone https://github.com/miftahstudy/idsc2026-glaucoma-detection.git
cd idsc2026-glaucoma-detection

pip install -r requirements.txt

python src/split.py
python src/train.py
python src/eval.py

streamlit run app.py

## 📊 Dataset
Dataset: Hillel Yaffe Glaucoma Dataset (HYGD) v1.0.0
Source: PhysioNet
Images: 747
Patients: 288
Labels: GON+ / GON−

Labels are based on OCT, visual field tests, and clinical follow-up.

🏗️ System Design

The system transforms a classification model into a clinical decision support tool:

Patient-level data splitting (prevents leakage)
Deep learning classification model
Probability-based risk scoring
Grad-CAM interpretability
Threshold-based triage system
🔒 Data Splitting Strategy

A strict patient-level split is used:

prevents data leakage
ensures realistic evaluation
avoids same-patient bias
🤖 Model Architecture
Backbone: ResNet18 (ImageNet pretrained)
Output: Binary classification (GON+ / GON−)
Activation: Sigmoid

Chosen for:

stability
efficiency
deployability
⚙️ Training Strategy
Loss: Binary Cross Entropy with Logits
Optimizer: Adam
Lightweight training setup (CPU/GPU friendly)
📈 Performance

Evaluated on a held-out patient-level test set:

AUC: 0.9940
Accuracy: 0.9434
Sensitivity: 0.9615
Specificity: 0.8929
F1-score: 0.9615

High sensitivity ensures safe screening performance.

🔍 Interpretability (Grad-CAM)

Grad-CAM is used to visualize model attention:

GON+ → activation near optic disc
GON− → diffuse / minimal activation

Supports:

transparency
clinical trust
model validation
🧠 Risk Stratification
Category	Probability Range
High Risk	p ≥ 0.85
Moderate Risk	0.50 ≤ p < 0.85
Borderline	0.35 ≤ p < 0.50
Low Risk	p < 0.35
⚠️ Uncertainty Handling
Default threshold: 0.5
Adjustable for sensitivity vs specificity

Borderline predictions:

require manual review
may require re-imaging
🏥 Clinical Workflow Integration
Fundus image captured
Model predicts probability
Patient categorized:
High-risk → urgent referral
Borderline → further review
Low-risk → routine monitoring
📊 Streamlit App Features
🔍 Prediction
Upload retinal image
Probability prediction
Risk category
Clinical recommendation
Grad-CAM visualization
Quality warnings
📈 EDA & Insights
Dataset distribution
Patient statistics
Image quality analysis
Sample visualization
🧪 Quality-Aware Warnings

Heuristic image quality checks:

Blur (Laplacian variance)
Brightness (mean intensity)
Contrast (std deviation)

Categories:

Good
Fair
Poor

⚠️ Not clinically validated — used as support only.

📂 Repository Structure
.
├── app.py
├── README.md
├── requirements.txt
│
├── data/
├── splits/
├── models/
├── outputs/
│
└── src/
🔁 Reproducibility
python src/split.py
python src/train.py
python src/eval.py
python src/gradcam.py
⚠️ Limitations
Small dataset (747 images)
Single-center data
Binary classification only
No external validation
Heuristic quality module
🌟 Design Philosophy

Practical, interpretable, and deployable

This system prioritizes:

robustness
simplicity
real-world usability
📚 Citation
Dataset (HYGD)

Abramovich et al., 2025
https://doi.org/10.13026/z0ak-km33

PhysioNet

Goldberger et al., 2000

⚠️ Disclaimer

This project is for research and competition purposes only.
It is not a medical diagnostic tool and should not replace professional clinical judgment.
