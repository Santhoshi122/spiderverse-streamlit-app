import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Page settings
st.set_page_config(page_title="Spider-Verse Character Guess", page_icon="🕸️")

st.title("🕸️ Spider-Verse Character Guess")
st.write("Upload an image and the AI will guess the Spider-Verse character.")

# Load model only once (faster)
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# File uploader
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:

    image = Image.open(uploaded)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Running prediction...")

    results = model(image)[0]

    # Get prediction
    name = results.names[results.probs.top1]
    confidence = results.probs.top1conf.item() * 100

    st.success(f"Prediction: {name}")
    st.write(f"Confidence: {confidence:.2f}%")
