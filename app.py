import streamlit as st

from utils.model_loader import load_model
from utils.predict import predict_image

st.set_page_config(
    page_title="Papaya Leaf Disease Detection",
    layout="wide"
)

st.title("🍃 Papaya Leaf Disease Detection")

model = load_model(
    "model/papayaleaf_cnn_5class.pth"
)

uploaded_file = st.file_uploader(
    "Upload gambar daun pepaya",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    with open("uploads/temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(
        "uploads/temp.jpg",
        caption="Uploaded Image",
        use_container_width=True
    )

    result = predict_image(
        model,
        "uploads/temp.jpg"
    )

    st.success(
        f"Prediksi: {result['class']}"
    )

    st.info(
        f"Confidence: {result['confidence']:.2f}%"
    )