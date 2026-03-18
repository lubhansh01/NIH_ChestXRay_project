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

    st.subheader("🔍 Top 5 Predictions") 

    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    
    for idx in top_5_indices:
        st.write(f"**{DISEASE_LABELS[idx]}** : {probabilities[idx]:.2f}")

    # --------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------
    st.subheader("🧠 Explainability (Grad-CAM)")

    target_layer = model.model.features[-1]
    gradcam = GradCAM(model, target_layer)

    top_class = top_5_indices[0]
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
    st.subheader("📚 AI Explanation (Top 5 - RAG grounded)")
    
    for idx in top_5_indices:
        label = DISEASE_LABELS[idx]
        prob = probabilities[idx]
        
        if label in KNOWLEDGE_BASE:
            st.markdown(f"**{label} ({prob:.2f})**: {KNOWLEDGE_BASE[label]}")
            
        else:
            st.markdown(
                f"**{label} ({prob:.2f})**: No detailed knowledge available. Clinical correlation recommended."
                )