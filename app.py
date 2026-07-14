import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Sign Language Recognition")
st.write(
    "Upload an image of an American Sign Language (ASL) hand gesture "
    "to predict the corresponding sign."
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# -----------------------------
# Class Labels
# -----------------------------
# IMPORTANT:
# Update this list if your dataset/classes are different.
class_names = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y",
    "del", "nothing", "space"
]

# Check model output
num_classes = model.output_shape[-1]

if len(class_names) != num_classes:
    st.error(
        f"""
Model output classes = {num_classes}

But class_names contains {len(class_names)} labels.

Please update class_names so that both numbers are equal.
"""
    )
    st.stop()

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((64, 64))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        input_image = preprocess_image(image)

        prediction = model.predict(input_image)

        predicted_index = np.argmax(prediction)

        predicted_label = class_names[predicted_index]

        confidence = float(prediction[0][predicted_index])

        st.success(f"Prediction: **{predicted_label}**")

        st.info(f"Confidence: **{confidence*100:.2f}%**")

        st.subheader("Prediction Probabilities")

        probabilities = {
            class_names[i]: float(prediction[0][i])
            for i in range(num_classes)
        }

        st.bar_chart(probabilities)

st.markdown("---")
st.caption("Developed using TensorFlow and Streamlit")