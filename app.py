import streamlit as st

st.set_page_config(
    page_title="Papaya Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Papaya Leaf Disease Detection")

uploaded_file = st.file_uploader(
    "Upload gambar daun pepaya",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Gambar yang diupload")