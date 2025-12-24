import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Page config
st.set_page_config(page_title="Traffic Sign Recognition", layout="centered")

st.title("🚦 Traffic Sign Recognition App")
st.write("Upload a traffic sign image and the CNN model will predict the sign.")

# Google Drive model link (FIXED)
MODEL_URL = "https://drive.google.com/uc?id=1IsBhzfN6qzSHwfl-F7u6GCO6wRk533mj"
MODEL_PATH = "traffic_sign_cnn.h5"

# Download model if not exists
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

IMG_SIZE = (64, 64)

# GTSRB class mapping
gtsrb_classes = {
  0: "Speed limit (20km/h)",
  1: "Speed limit (30km/h)",
  2: "Speed limit (50km/h)",
  3: "Speed limit (60km/h)",
  4: "Speed limit (70km/h)",
  5: "Speed limit (80km/h)",
  6: "End of speed limit (80km/h)",
  7: "Speed limit (100km/h)",
  8: "Speed limit (120km/h)",
  9: "No passing",
  10: "No passing for vehicles > 3.5 tons",
  11: "Right-of-way at the next intersection",
  12: "Priority road",
  13: "Yield",
  14: "Stop",
  15: "No vehicles",
  16: "Vehicles > 3.5 tons prohibited",
  17: "No entry",
  18: "General caution",
  19: "Dangerous curve to the left",
  20: "Dangerous curve to the right",
  21: "Double curve",
  22: "Bumpy road",
  23: "Slippery road",
  24: "Road narrows on the right",
  25: "Road work",
  26: "Traffic signals",
  27: "Pedestrians",
  28: "Children crossing",
  29: "Bicycles crossing",
  30: "Beware of ice/snow",
  31: "Wild animals crossing",
  32: "End of all speed and passing limits",
  33: "Turn right ahead",
  34: "Turn left ahead",
  35: "Ahead only",
  36: "Go straight or right",
  37: "Go straight or left",
  38: "Keep right",
  39: "Keep left",
  40: "Roundabout mandatory",
  41: "End of no passing",
  42: "End of no passing for vehicles > 3.5 tons"
}

uploaded_file = st.file_uploader("📤 Upload Traffic Sign Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"🛑 Predicted Sign: **{gtsrb_classes[class_id]}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")

