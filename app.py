import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from sklearn.neighbors import KNeighborsClassifier

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Absensi Face ID", layout="wide")
st.title("Sistem Absensi Berbasis Face ID (Mediapipe)")
st.write("Aplikasi ini menggunakan deteksi wajah Mediapipe untuk mencatat kehadiran.")

# --- PATH PENYIMPANAN ---
ENCODINGS_PATH = 'face_encodings.pkl'
ATTENDANCE_PATH = 'attendance.csv'

# --- INISIALISASI MEDIAPIPE ---
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils


# --- FUNGSI BANTU ---
def load_known_faces():
    try:
        with open(ENCODINGS_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {"names": [], "embeddings": []}


def save_known_faces(data):
    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump(data, f)


def log_attendance(name):
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Nama', 'Waktu'])

    if not df.empty:
        last_entry = df[df['Nama'] == name]
        if not last_entry.empty:
            last_time = datetime.strptime(last_entry['Waktu'].iloc[-1], '%Y-%m-%d %H:%M:%S')
            if (datetime.now() - last_time).total_seconds() < 60:
                return

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df = pd.concat([df, pd.DataFrame([[name, now]], columns=['Nama', 'Waktu'])], ignore_index=True)
    df.to_csv(ATTENDANCE_PATH, index=False)


def get_embedding(img):
    """Ambil fitur wajah sederhana (bounding box + keypoints)"""
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.detections:
            det = results.detections[0]
            box = det.location_data.relative_bounding_box
            keypoints = []
            for kp in det.location_data.relative_keypoints:
                keypoints.extend([kp.x, kp.y])
            emb = np.array([box.xmin, box.ymin, box.width, box.height] + keypoints)
            return emb
    return None


# --- MUAT DATA WAJAH ---
faces_data = load_known_faces()


# --- SIDEBAR MODE ---
st.sidebar.header("Mode Aplikasi")
app_mode = st.sidebar.selectbox("Pilih Mode", ["Pendaftaran Wajah", "Absensi Real-time"])


# --- MODE PENDAFTARAN WAJAH ---
if app_mode == "Pendaftaran Wajah":
    st.header("Form Pendaftaran Wajah Baru")
    new_name = st.text_input("Masukkan Nama Anda:")

    img_file_buffer = st.camera_input("Ambil Foto Wajah (Pastikan wajah jelas)")

    if img_file_buffer and new_name:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        emb = get_embedding(cv2_img)
        if emb is not None:
            faces_data["names"].append(new_name)
            faces_data["embeddings"].append(emb)
            save_known_faces(faces_data)
            st.success(f"Wajah '{new_name}' berhasil disimpan!")
        else:
            st.error("Tidak ada wajah terdeteksi. Coba lagi.")


# --- MODE ABSENSI REAL-TIME ---
elif app_mode == "Absensi Real-time":
    st.header("Absensi Menggunakan Kamera")

    # Siapkan model KNN
    clf = None
    if faces_data["names"]:
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(faces_data["embeddings"], faces_data["names"])

    class FaceRecognitionTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
                results = fd.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                if results.detections:
                    for det in results.detections:
                        box = det.location_data.relative_bounding_box
                        h, w, _ = img.shape
                        x, y, ww, hh = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)

                        # Gambar kotak
                        cv2.rectangle(img, (x, y), (x + ww, y + hh), (0, 255, 0), 2)

                        # Ambil embedding
                        emb = get_embedding(img)
                        name = "Unknown"
                        if clf and emb is not None:
                            name = clf.predict([emb])[0]
                            log_attendance(name)

                        # Tulis nama
                        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            return img

    webrtc_streamer(key="absensi", video_transformer_factory=FaceRecognitionTransformer)

    st.subheader("Laporan Kehadiran")
    try:
        attendance_df = pd.read_csv(ATTENDANCE_PATH)
        st.dataframe(attendance_df.sort_values(by='Waktu', ascending=False), use_container_width=True)
    except FileNotFoundError:
        st.info("Belum ada data kehadiran yang tercatat.")
