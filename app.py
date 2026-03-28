import os
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
MODEL_PATH = os.path.join("models", "best.pth")
LABELS_PATH = os.path.join("data", "Labels.csv")
IMAGE_DIR = os.path.join("data", "Images")
TRAIN_SPLIT_PATH = os.path.join("splits", "train.csv")
VAL_SPLIT_PATH = os.path.join("splits", "val.csv")
TEST_SPLIT_PATH = os.path.join("splits", "test.csv")

st.set_page_config(
    page_title="Glaucoma Screening Support System",
    page_icon="👁️",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.card {
    background-color: #f8fafc;
    padding: 1rem 1.2rem;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}
.title-box {
    background: linear-gradient(135deg, #dbeafe, #eff6ff);
    padding: 1.2rem 1.4rem;
    border-radius: 18px;
    border: 1px solid #bfdbfe;
    margin-bottom: 1rem;
}
.small-muted {
    color: #475569;
    font-size: 0.95rem;
}
.footer-note {
    font-size: 0.85rem;
    color: #64748b;
}
.insight-box {
    background-color: #f8fafc;
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_labels():
    if os.path.exists(LABELS_PATH):
        return pd.read_csv(LABELS_PATH)
    return None

@st.cache_data
def load_split_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def get_sample_image_path(df, label_value):
    if df is None:
        return None
    subset = df[df["Label"] == label_value]
    if len(subset) == 0:
        return None
    img_name = subset.iloc[0]["Image Name"]
    img_path = os.path.join(IMAGE_DIR, img_name)
    return img_path if os.path.exists(img_path) else None

model = load_model()
labels_df = load_labels()
train_df = load_split_csv(TRAIN_SPLIT_PATH)
val_df = load_split_csv(VAL_SPLIT_PATH)
test_df = load_split_csv(TEST_SPLIT_PATH)

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =========================
# GRAD-CAM
# =========================
def generate_gradcam(pil_img, model):
    target_layer = model.layer4[-1]

    gradients = []
    activations = []

    def forward_hook(module, input_tensor, output_tensor):
        activations.append(output_tensor)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_resized)

    x = transform(pil_img).unsqueeze(0).to(DEVICE)

    model.zero_grad()
    out = model(x)
    prob = torch.sigmoid(out)[0, 0].item()
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
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fh.remove()
    bh.remove()

    return prob, overlay_rgb

# =========================
# HELPERS
# =========================
def get_prediction_text(prob, threshold=0.5):
    return "GON+" if prob >= threshold else "GON-"

def get_risk_label(prob, threshold=0.5):
    if prob >= 0.85:
        return "High Risk"
    elif prob >= threshold:
        return "Moderate Risk"
    elif prob >= 0.35:
        return "Borderline"
    else:
        return "Low Risk"

def get_clinical_note(prob, threshold=0.5):
    if prob >= 0.85:
        return (
            "Prioritize ophthalmology review. The model detects a strong glaucomatous pattern."
        )
    elif prob >= threshold:
        return (
            "Further clinical assessment is recommended. The image is predicted as glaucomatous."
        )
    elif prob >= 0.35:
        return (
            "Prediction is near the decision boundary. Consider repeat acquisition or manual review."
        )
    else:
        return (
            "Low-risk output from the model. This does not replace formal clinical examination."
        )

def risk_color(prob, threshold=0.5):
    if prob >= 0.85:
        return "error"
    elif prob >= threshold:
        return "warning"
    elif prob >= 0.35:
        return "warning"
    else:
        return "success"

def make_bar_chart(series, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(6, 4))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig

def make_hist_chart(values, title, xlabel, ylabel, bins=10):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig

def get_split_summary(df, name):
    if df is None:
        return {
            "Split": name,
            "Images": "-",
            "Patients": "-",
            "GON+": "-",
            "GON-": "-"
        }
    return {
        "Split": name,
        "Images": len(df),
        "Patients": df["Patient"].nunique(),
        "GON+": int((df["Label"] == "GON+").sum()),
        "GON-": int((df["Label"] == "GON-").sum())
    }

def assess_image_quality(pil_img):
    """
    Simple heuristic quality check:
    - blur: variance of Laplacian
    - brightness: mean grayscale intensity
    - contrast: std grayscale intensity
    """
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    contrast = gray.std()

    warnings = []
    quality_level = "Good"

    # Conservative heuristic thresholds
    if blur_score < 40:
        warnings.append("Image may be blurry.")
    if brightness < 45:
        warnings.append("Image appears too dark.")
    elif brightness > 210:
        warnings.append("Image appears too bright.")
    if contrast < 25:
        warnings.append("Image contrast appears low.")

    if len(warnings) >= 2:
        quality_level = "Poor"
    elif len(warnings) == 1:
        quality_level = "Fair"

    return {
        "quality_level": quality_level,
        "blur_score": float(blur_score),
        "brightness": float(brightness),
        "contrast": float(contrast),
        "warnings": warnings
    }

def render_quality_warning(quality_result, prob, threshold):
    quality_level = quality_result["quality_level"]
    warnings = quality_result["warnings"]

    borderline = (0.35 <= prob < threshold) or (abs(prob - threshold) < 0.10)

    if quality_level == "Poor":
        st.error(
            "Image quality warning: this fundus image may be too blurry, too dark/bright, or too low in contrast "
            "for reliable screening. Repeat acquisition is recommended."
        )
    elif quality_level == "Fair":
        st.warning(
            "Image quality caution: the uploaded image may be suboptimal. "
            "Interpret the prediction carefully and consider reacquisition if clinically needed."
        )

    if warnings:
        st.markdown("**Detected quality issues:**")
        for w in warnings:
            st.markdown(f"- {w}")

        st.caption(
            "Note: This is a heuristic-based quality assessment and does not replace clinically validated quality scoring methods."
        )

    if borderline:
        st.info(
            "Prediction uncertainty notice: this case is near the decision boundary. "
            "Manual review or repeat imaging is recommended."
        )

# =========================
# HEADER
# =========================
st.markdown("""
<div class="title-box">
    <h1 style="margin-bottom:0.3rem;">👁️ Glaucoma Screening Support System</h1>
    <div class="small-muted">
        Deep learning–based decision support for retinal fundus image screening using HYGD v1.0.0
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
This demo is designed as a **triage-support tool** for glaucoma screening.  
It is **not** intended to replace ophthalmologists or standalone clinical diagnosis.
""")

# =========================
# TOP SUMMARY
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC", "0.9940")
c2.metric("Accuracy", "0.9434")
c3.metric("Sensitivity", "0.9615")
c4.metric("Specificity", "0.8929")

with st.expander("About this system"):
    st.write("""
- **Dataset:** HYGD v1.0.0 (PhysioNet)
- **Task:** Binary classification of glaucomatous optic neuropathy (GON+ vs GON-)
- **Model:** ResNet18
- **Evaluation setting:** Patient-level split
- **Interpretability:** Grad-CAM
- **Clinical framing:** Screening-support / triage-support
""")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("System Controls")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.markdown("---")
    st.subheader("Clinical Safety Notes")
    st.write("""
- High sensitivity is prioritized for screening.
- Borderline outputs should be reviewed cautiously.
- Low-quality or ambiguous images may require reacquisition.
""")

    st.markdown("---")
    st.subheader("Recommended Demo Flow")
    st.write("""
1. Upload fundus image  
2. Review prediction probability  
3. Inspect Grad-CAM attention  
4. Read triage recommendation  
5. Review image quality warning
""")

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["👁️ Prediction", "📊 EDA & Insights"])

# =========================
# TAB 1: PREDICTION
# =========================
with tab1:
    uploaded_file = st.file_uploader(
        "Upload a retinal fundus image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Upload a retinal fundus image to run screening prediction and Grad-CAM visualization.")

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        quality_result = assess_image_quality(pil_img)

        with st.spinner("Running prediction and Grad-CAM..."):
            prob, gradcam_img = generate_gradcam(pil_img, model)
            pred = get_prediction_text(prob, threshold)
            risk = get_risk_label(prob, threshold)
            note = get_clinical_note(prob, threshold)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Uploaded Image")
            st.image(pil_img, width=450)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Grad-CAM Visualization")
            st.image(gradcam_img, width=450)
            st.caption("Highlighted regions indicate image areas that contributed strongly to the model prediction.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Prediction Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", pred)
        m2.metric("Probability of GON+", f"{prob:.3f}")
        m3.metric("Risk Category", risk)

        st.markdown("### Clinical Recommendation")
        color_mode = risk_color(prob, threshold)
        if color_mode == "error":
            st.error(note)
        elif color_mode == "warning":
            st.warning(note)
        else:
            st.success(note)

        st.markdown("### Image Quality Heuristic Check")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Quality Level", quality_result["quality_level"])
        q2.metric("Blur Score", f"{quality_result['blur_score']:.1f}")
        q3.metric("Brightness", f"{quality_result['brightness']:.1f}")
        q4.metric("Contrast", f"{quality_result['contrast']:.1f}")

        render_quality_warning(quality_result, prob, threshold)

        st.markdown("### Interpretation Notes")
        st.markdown("""
- The model predicts **GON+** when the estimated probability exceeds the chosen threshold.
- The heatmap should ideally concentrate near clinically relevant structures such as the **optic disc region**.
- Borderline cases should be interpreted cautiously and may benefit from repeat image acquisition or manual review.
- Image quality warnings provide an additional safety layer during screening.
- This tool is for **decision support only**, not definitive diagnosis.
""")

# =========================
# TAB 2: EDA
# =========================
with tab2:
    st.subheader("Dataset Overview")

    if labels_df is None:
        st.error("Labels.csv not found at data/Labels.csv")
    else:
        total_images = len(labels_df)
        total_patients = labels_df["Patient"].nunique()
        gon_pos = (labels_df["Label"] == "GON+").sum()
        gon_neg = (labels_df["Label"] == "GON-").sum()

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Total Images", total_images)
        o2.metric("Total Patients", total_patients)
        o3.metric("GON+", gon_pos)
        o4.metric("GON-", gon_neg)

        st.markdown("### Exploratory Data Analysis")

        label_counts = labels_df["Label"].value_counts().reindex(["GON+", "GON-"]).fillna(0)
        fig1 = make_bar_chart(
            label_counts,
            "Label Distribution",
            "Label",
            "Count"
        )

        images_per_patient = labels_df.groupby("Patient").size()
        fig2 = make_hist_chart(
            images_per_patient.values,
            "Images per Patient Distribution",
            "Number of Images per Patient",
            "Frequency",
            bins=10
        )

        quality_scores = labels_df["Quality Score"].dropna()
        fig3 = make_hist_chart(
            quality_scores.values,
            "Quality Score Distribution",
            "Quality Score",
            "Frequency",
            bins=10
        )

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(fig1)
        with c2:
            st.pyplot(fig2)

        st.pyplot(fig3)

        st.markdown("### Sample Fundus Images")
        sample_gon_pos = get_sample_image_path(labels_df, "GON+")
        sample_gon_neg = get_sample_image_path(labels_df, "GON-")

        s1, s2 = st.columns(2)
        with s1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Sample GON+")
            if sample_gon_pos:
                st.image(sample_gon_pos, width=350)
            else:
                st.info("Sample GON+ image not found.")
            st.markdown('</div>', unsafe_allow_html=True)

        with s2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Sample GON-")
            if sample_gon_neg:
                st.image(sample_gon_neg, width=350)
            else:
                st.info("Sample GON- image not found.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Split Summary")
        split_summary = pd.DataFrame([
            get_split_summary(train_df, "Train"),
            get_split_summary(val_df, "Validation"),
            get_split_summary(test_df, "Test")
        ])
        st.dataframe(split_summary, width="stretch")

        st.markdown("### Data Insights")

        gon_pos_pct = (gon_pos / total_images) * 100
        gon_neg_pct = (gon_neg / total_images) * 100
        avg_img_per_patient = images_per_patient.mean()
        min_q = quality_scores.min() if len(quality_scores) > 0 else None
        max_q = quality_scores.max() if len(quality_scores) > 0 else None
        mean_q = quality_scores.mean() if len(quality_scores) > 0 else None

        st.markdown(f"""
<div class="insight-box">
<b>1. Class Imbalance</b><br>
The dataset is imbalanced, with approximately <b>{gon_pos_pct:.1f}% GON+</b> and <b>{gon_neg_pct:.1f}% GON-</b>. 
This is important because a screening model should not rely only on accuracy; it must also maintain strong sensitivity to avoid missing glaucoma cases.
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="insight-box">
<b>2. Patient-Level Structure</b><br>
Each patient contributes an average of <b>{avg_img_per_patient:.2f} images</b>. 
This confirms that patient-level splitting is necessary to prevent data leakage, because image-level splitting could place the same patient in both training and testing sets.
</div>
""", unsafe_allow_html=True)

        if mean_q is not None:
            st.markdown(f"""
<div class="insight-box">
<b>3. Image Quality Variability</b><br>
Quality scores range from <b>{min_q:.1f}</b> to <b>{max_q:.1f}</b>, with a mean of <b>{mean_q:.2f}</b>. 
This indicates that the dataset reflects real-world variability rather than perfectly curated data, which strengthens the relevance of the system for practical screening deployment.
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="insight-box">
<b>4. Split Integrity</b><br>
The train, validation, and test sets are separated at the <b>patient level</b>, not at the image level. 
This makes the reported performance more trustworthy because the model is evaluated on previously unseen patients.
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="insight-box">
<b>5. Why This Matters Clinically</b><br>
Because glaucoma can be asymptomatic in early stages, a dataset like this supports development of a model that is not only accurate, but also useful for <b>risk-based screening and triage</b>. 
The EDA shows that the problem is clinically meaningful, imbalanced, and patient-structured — exactly the kind of scenario where careful validation is essential.
</div>
""", unsafe_allow_html=True)

        st.markdown("### Why EDA Matters for This Project")
        st.markdown("""
- It demonstrates that the dataset is **patient-structured**, so leakage prevention is necessary.
- It shows that the dataset is **class-imbalanced**, so sensitivity matters more than raw accuracy.
- It reveals **quality variability**, which supports the need for cautious interpretation in real-world screening.
- It strengthens the mathematical and clinical rigor of the overall system.
""")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div class="footer-note">
For research and competition demonstration purposes only.  
This system should not be used as a standalone diagnostic device.
</div>
""", unsafe_allow_html=True)