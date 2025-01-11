import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import tensorflow as tf

# Load Model
model = tf.keras.models.load_model("model/final_asl_model.keras")
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["nothing", "space", "del"]

# Fungsi untuk prediksi
def predict_sign(image):
    image = cv2.resize(image, (224, 224))  # Resize sesuai model
    image = np.expand_dims(image, axis=0) / 255.0  # Normalisasi
    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    return labels[class_idx]

# Konfigurasi Halaman
st.set_page_config(
    page_title="Future Sign Language Translator",
    page_icon="🌟",
    layout="wide"
)

# **CSS** untuk Background Gradient dan Gaya Tambahan
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: white;
        }
        .block-container {
            padding-top: 2rem;
        }
        .stButton > button {
            background-color: #6a11cb;
            color: white;
            border-radius: 12px;
            border: none;
            padding: 0.5rem 1.5rem;
        }
        .stButton > button:hover {
            background-color: #2575fc;
            color: white;
        }
        .gallery-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            margin-top: 10px;
        }
        .gallery-image {
            border: 2px solid white;
            border-radius: 8px;
            width: 80px;
            height: 80px;
            object-fit: cover;
        }
    </style>
""", unsafe_allow_html=True)

# **Judul Aplikasi**
st.title("🌟 Future Sign Language Translator 🌟")

# **Sidebar dengan Logo**
with st.sidebar:
    st.image("static/img/logo.webp", use_column_width=True)  # Logo
    st.header("👋 Welcome!")
    st.markdown("""
    - **Start Camera**: Untuk memulai kamera.
    - **Stop Camera**: Untuk menghentikan kamera.
    - **Upload Image**: Unggah gambar untuk prediksi.
    """)
    st.markdown("---")
    st.markdown("📚 **Example Sign Language Images**")

    # **Gallery Section di Sidebar**
    img_folder = "static/img/example"  # Lokasi folder gambar
    if os.path.exists(img_folder):
        img_files = sorted(
            [f for f in os.listdir(img_folder) if f.endswith(("jpg", "png", "jpeg"))]
        )
        if img_files:
            gallery_html = '<div class="gallery-container">'
            for img_file in img_files:
                img_path = os.path.join(img_folder, img_file).replace("\\", "/")  # Jalur relatif untuk HTML
                gallery_html += f'<img src="{img_path}" alt="{img_file.split(".")[0]}" class="gallery-image">'
            gallery_html += '</div>'
            st.markdown(gallery_html, unsafe_allow_html=True)
        else:
            st.warning("No example images found in the folder!")
    else:
        st.error(f"The folder '{img_folder}' does not exist!")

    st.markdown("---")
    st.markdown("📚 **Info**")
    st.info("Aplikasi ini menggunakan Machine Learning untuk menerjemahkan bahasa isyarat secara real-time.")

# **Video Feed & Prediksi**
col1, col2 = st.columns(2)

# Tombol untuk mengontrol kamera
camera = cv2.VideoCapture(0)
start_button = st.button("🔴 Start Camera")
stop_button = st.button("🛑 Stop Camera")

if start_button:
    st.session_state["camera_active"] = True
if stop_button:
    st.session_state["camera_active"] = False

if "camera_active" not in st.session_state:
    st.session_state["camera_active"] = False

# **Video Feed Section**
with col1:
    st.subheader("📹 Live Video Feed")
    stframe = st.empty()

# **Prediksi Section**
with col2:
    st.subheader("🧠 Predicted Sign")
    prediction_placeholder = st.empty()

# Kamera aktif
if st.session_state["camera_active"]:
    fps = 0
    start_time = time.time()
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Kamera tidak terdeteksi!")
            break

        # Prediksi tanda menggunakan model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sign_predicted = predict_sign(frame_rgb)

        # Tampilkan frame di aplikasi
        stframe.image(frame_rgb, channels="RGB", width=640)

        # Update prediksi
        prediction_placeholder.write(f"**Predicted Sign:** {sign_predicted}")

        # Hentikan loop saat tombol Stop diklik
        if not st.session_state["camera_active"]:
            break

# Pastikan kamera dilepaskan
camera.release()
cv2.destroyAllWindows()

# **Image Uploader Section**
st.markdown("---")
st.subheader("📤 Upload Image for Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    if st.button("Predict Uploaded Image"):
        # Konversi gambar ke format numpy array
        img_array = np.array(image)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB) if img_array.shape[-1] == 4 else img_array
        prediction = predict_sign(img_rgb)

        # Tampilkan prediksi
        st.success(f"Predicted Sign: **{prediction}**")

# **Footer**
st.markdown("""
    <hr>
    <center>🌍 Created for Inclusive Education | Powered by AI 🌟</center>
""", unsafe_allow_html=True)
