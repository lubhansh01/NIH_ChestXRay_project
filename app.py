import streamlit as st
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

from model import ChestXrayModel
from labels import DISEASE_LABELS
from gradcam import GradCAM
from rag_knowledge import KNOWLEDGE_BASE, DISCLAIMER


# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(
    page_title="NIH Chest X-ray AI",
    layout="wide"
)

st.title("🫁 NIH Chest X-ray14 AI Assistant")
st.markdown(DISCLAIMER)


# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = ChestXrayModel(num_classes=14)
    model.eval()
    return model

model = load_model()


# --------------------------------------------------
# Image preprocessing (RGB – 3 channels)
# --------------------------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# File upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a Chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image as RGB
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-ray", width=300)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------
    st.subheader("🔍 Model Predictions")

    for label, prob in zip(DISEASE_LABELS, probabilities):
        st.write(f"**{label}** : {prob:.2f}")

    # --------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------
    st.subheader("🧠 Explainability (Grad-CAM)")

    target_layer = model.model.features[-1]
    gradcam = GradCAM(model, target_layer)

    top_class = int(np.argmax(probabilities))
    cam = gradcam.generate(input_tensor, top_class)

    img_resized = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
    st.image(overlay, caption=f"Grad-CAM: {DISEASE_LABELS[top_class]}")

    # --------------------------------------------------
    # RAG-grounded explanation
    # --------------------------------------------------
    st.subheader("📚 AI Explanation (RAG-grounded)")

    found = False
    for label, prob in zip(DISEASE_LABELS, probabilities):
        if prob > 0.5 and label in KNOWLEDGE_BASE:
            st.markdown(f"**{label}**: {KNOWLEDGE_BASE[label]}")
            found = True

    if not found:
        st.info(
            "No high-confidence abnormal findings detected. "
            "Clinical correlation is recommended."
        )
