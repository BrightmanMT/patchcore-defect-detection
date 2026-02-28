import streamlit as st
import torch
import numpy as np
import pandas as pd
import time
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import torchvision.models as models

# Define the PatchCoreModel class
class PatchCoreModel(torch.nn.Module):
    def __init__(self):
        super(PatchCoreModel, self).__init__()
        # Use a pre-trained ResNet18 as the feature extractor backbone
        # Initialize with pre-trained ImageNet weights to ensure all layers are present
        # and initialized. This will allow partial loading of state_dict later.
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # PatchCore typically extracts features from specific layers.
        # We'll define a forward hook or adapt the forward pass for this.
        # For now, let's assume the checkpoint's 'model' key contains the full ResNet.
        # If only features were saved, this might need further refinement.

    def forward(self, x):
        # Define forward pass based on how features are extracted in PatchCore
        # This is a basic forward pass for a ResNet. Actual PatchCore might need
        # to extract features from intermediate layers.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # For PatchCore, we usually don't need the final classification layer
        # and might return features from earlier stages. For simplicity, this
        # placeholder returns the output after layer4, but this might need
        # to be adjusted based on the actual PatchCore implementation.
        return x # Placeholder output, adjust based on PatchCore requirements

# Placeholder for compute_anomaly function (replace with your actual implementation)
def compute_anomaly(model, image_tensor):
    # This is a placeholder. Replace with actual anomaly computation.
    # Example: run image through model, compute anomaly score and map.
    dummy_anomaly_map = np.random.rand(224, 224) # Example heatmap
    dummy_score = np.random.rand() * 0.5 + 0.5 # Example score between 0.5 and 1.0
    return dummy_anomaly_map, dummy_score


# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="Industrial Defect Detection",
    layout="wide"
)

DEVICE = "cpu"  # Streamlit Cloud / local CPU

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load("/content/drive/MyDrive/Colab Notebooks/patchcore_streamlit/patchcore_bottle.pt", map_location=DEVICE, weights_only=False)

    # Instantiate the PatchCoreModel with the correct architecture
    model = PatchCoreModel()
    # Load the ResNet state_dict into the backbone of the PatchCoreModel
    # Use strict=False to allow loading a partial state_dict (e.g., if fc layer or later layers are missing)
    model.backbone.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    threshold = checkpoint["threshold"]
    return model, threshold

model, THRESHOLD = load_model()

# -------------------------------
# UTILITIES
# -------------------------------
def preprocess(image: Image.Image):
    img = np.array(image.resize((224, 224)))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img

def overlay_heatmap(image, heatmap, alpha):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def inspect_image(image_tensor):
    start = time.time()
    with torch.no_grad():
        anomaly_map, score = compute_anomaly(model, image_tensor)
    fps = 1 / (time.time() - start)
    return anomaly_map, score, fps

# -------------------------------
# UI HEADER
# -------------------------------
st.title("🏭 Industrial Defect Detection System")
st.markdown(
    "**PatchCore-based anomaly detection for visual inspection**  \n"
    "Designed for factory-style OK / NG decision making"
)

st.divider()

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("Inspection Settings")

alpha = st.sidebar.slider(
    "Heatmap Transparency",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

st.sidebar.markdown(f"**Decision Threshold:** `{THRESHOLD:.3f}`")

# -------------------------------
# MAIN TABS
# -------------------------------
tab1, tab2 = st.tabs(["🔍 Single Image Inspection", "📂 Batch Inspection"])

inspection_log = []

# ===============================
# TAB 1 — SINGLE IMAGE
# ===============================
with tab1:
    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img_tensor = preprocess(image).unsqueeze(0)

        anomaly_map, score, fps = inspect_image(img_tensor)

        decision = "NG (Defect)" if score >= THRESHOLD else "OK (Normal)"
        # color = "red" if decision.startswith("NG") else "green"

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Inspection Result")
            # Visual Decision Badge
            if decision.startswith("NG"):
                st.error("❌ NG — Defect Detected")
            else:
                st.success("✅ OK — Normal Product")

            st.write(f"**Anomaly Score:** `{score:.4f}`")

            # Confidence Meter
            confidence = min(score / THRESHOLD, 1.0)
            st.progress(confidence, text=f"Confidence: {confidence:.2f}")

            st.write(f"**FPS:** `{fps:.2f}`")

        img_np = np.array(image)
        overlay = overlay_heatmap(img_np, anomaly_map, alpha)

        st.subheader("Anomaly Heatmap Overlay")
        st.image(overlay, use_column_width=True)
        st.caption("Heatmap: Red = high anomaly, Blue = normal")

        # Save Inspection History
        st.session_state.history.append({
            "score": score,
            "decision": decision,
            "fps": fps,
            "filename": uploaded.name # Added filename for better history
        })

        st.subheader("Inspection History")
        st.dataframe(pd.DataFrame(st.session_state.history))

# ===============================
# TAB 2 — BATCH MODE
# ===============================
with tab2:
    st.markdown("### Upload multiple images for batch inspection")

    files = st.file_uploader(
        "Upload images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if files:
        progress = st.progress(0)
        results = []

        for idx, file in enumerate(files):
            image = Image.open(file).convert("RGB")
            img_tensor = preprocess(image).unsqueeze(0)

            anomaly_map, score, fps = inspect_image(img_tensor)
            decision = "NG" if score >= THRESHOLD else "OK"

            results.append({
                "filename": file.name,
                "score": score,
                "decision": decision,
                "fps": fps
            })

            progress.progress((idx + 1) / len(files))

        df = pd.DataFrame(results)

        st.subheader("Batch Inspection Results")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV Report",
            csv,
            "inspection_results.csv",
            "text/csv"
        )

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.markdown(
    "🔧 **System Notes**  \n"
    "- Model trained on normal samples only (unsupervised anomaly detection)  \n"
    "- Patch-based feature distance using pretrained CNN  \n"
    "- Designed for proof-of-concept factory deployment"
)
