import streamlit as st
import cv2
import face_recognition
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="Absensi Face ID", layout="wide")
st.title("Sistem Absensi Berbasis Face ID")
st.write("Aplikasi ini menggunakan pengenalan wajah untuk mencatat kehadiran.")

# --- INISIALISASI & FUNGSI BANTU ---

# Path untuk file penyimpanan
ENCODINGS_PATH = 'face_encodings.pkl'
ATTENDANCE_PATH = 'attendance.csv'

# Fungsi untuk memuat encoding wajah yang sudah tersimpan
def load_known_faces():
    try:
        with open(ENCODINGS_PATH, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
            return known_face_encodings, known_face_names
    except FileNotFoundError:
        return [], []

# Fungsi untuk menyimpan encoding wajah baru
def save_known_faces(known_face_encodings, known_face_names):
    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

# Fungsi untuk mencatat kehadiran
def log_attendance(name):
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Nama', 'Waktu'])

    # Cek apakah orang tersebut sudah absen dalam 1 menit terakhir untuk menghindari duplikasi
    if not df.empty:
        last_entry = df[df['Nama'] == name]
        if not last_entry.empty:
            last_time_str = last_entry['Waktu'].iloc[-1]
            last_time = datetime.strptime(last_time_str, '%Y-%m-%d %H:%M:%S')
            if (datetime.now() - last_time).total_seconds() < 60:
                return # Jangan catat jika kurang dari 60 detik

    now = datetime.now()
    current_time = now.strftime('%Y-%m-%d %H:%M:%S')
    new_entry = pd.DataFrame([[name, current_time]], columns=['Nama', 'Waktu'])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(ATTENDANCE_PATH, index=False)


# Muat data wajah yang sudah ada
known_face_encodings, known_face_names = load_known_faces()

# --- SIDEBAR & MODE APLIKASI ---
st.sidebar.header("Mode Aplikasi")
app_mode = st.sidebar.selectbox("Pilih Mode", ["Pendaftaran Wajah", "Absensi Real-time"])

# --- MODE 1: PENDAFTARAN WAJAH ---
if app_mode == "Pendaftaran Wajah":
    st.header("Form Pendaftaran Wajah Baru")
    
    new_name = st.text_input("Masukkan Nama Anda:")
    
    # Inisialisasi session state untuk menyimpan gambar yang diambil
    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = []

    # Menggunakan st.camera_input untuk mengambil gambar
    img_file_buffer = st.camera_input("Ambil 5 Foto Wajah (Pastikan wajah terlihat jelas dan pencahayaan baik)")

    if img_file_buffer is not None:
        # Konversi buffer gambar ke format yang bisa dibaca OpenCV
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        if st.button(f"Tambahkan Foto ke-{len(st.session_state.captured_images) + 1}"):
            st.session_state.captured_images.append(cv2_img)
            st.success(f"Foto ke-{len(st.session_state.captured_images)} berhasil ditambahkan.")

    st.write(f"Jumlah foto yang sudah diambil: **{len(st.session_state.captured_images)}/5**")

    # Tampilkan thumbnail gambar yang sudah diambil
    if st.session_state.captured_images:
        st.image([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in st.session_state.captured_images], width=150)

    if len(st.session_state.captured_images) == 5:
        if st.button("Proses dan Simpan Wajah"):
            if new_name:
                with st.spinner("Memproses gambar dan menyimpan encoding..."):
                    # Loop melalui setiap gambar yang diambil
                    for img in st.session_state.captured_images:
                        # Konversi ke RGB karena face_recognition menggunakan format ini
                        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Deteksi lokasi wajah
                        face_locations = face_recognition.face_locations(rgb_image)
                        
                        # Jika wajah ditemukan, dapatkan encodingnya
                        if face_locations:
                            # Asumsikan hanya ada satu wajah per gambar
                            face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
                            
                            # Tambahkan encoding dan nama ke list
                            known_face_encodings.append(face_encoding)
                            known_face_names.append(new_name)
                        else:
                            st.warning(f"Tidak ada wajah yang terdeteksi di salah satu gambar. Coba lagi.")
                            continue

                    # Simpan data yang sudah diperbarui
                    save_known_faces(known_face_encodings, known_face_names)
                    st.success(f"Wajah untuk '{new_name}' berhasil disimpan!")
                    st.session_state.captured_images = [] # Kosongkan setelah disimpan
            else:
                st.error("Nama tidak boleh kosong!")

# --- MODE 2: ABSENSI REAL-TIME ---
elif app_mode == "Absensi Real-time":
    st.header("Absensi Menggunakan Kamera")
    st.write("Arahkan wajah Anda ke kamera. Sistem akan mengenali dan mencatat kehadiran Anda secara otomatis.")

    class FaceRecognitionTransformer(VideoTransformerBase):
        def __init__(self):
            self.known_face_encodings, self.known_face_names = load_known_faces()

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Perkecil frame untuk pemrosesan lebih cepat
            small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Temukan semua wajah dan encodingnya di frame saat ini
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        log_attendance(name) # Catat kehadiran

                face_names.append(name)

            # Tampilkan hasil di frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Kembalikan ke ukuran semula karena frame diperkecil
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Gambar kotak di sekitar wajah
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

                # Tulis nama di bawah kotak
                cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            return img

    webrtc_streamer(key="absensi", video_transformer_factory=FaceRecognitionTransformer)
    
    st.subheader("Laporan Kehadiran")
    try:
        attendance_df = pd.read_csv(ATTENDANCE_PATH)
        st.dataframe(attendance_df.sort_values(by='Waktu', ascending=False), use_container_width=True)
    except FileNotFoundError:
        st.info("Belum ada data kehadiran yang tercatat.")