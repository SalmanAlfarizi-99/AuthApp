import os
import csv
import cv2
import random
import pickle
import numpy as np

from datetime import datetime
from keras_facenet import FaceNet
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity  # jika tidak dipakai, bisa dihapus

from services.evaluations import run_evaluation, save_prediction_log


# Router
auth_router = APIRouter(prefix="/auth", tags=["auth"])

# Load Model & Encoder
print("Loading FaceNet & Classifier at startup...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

embedder = FaceNet()

with open(os.path.join(BASE_DIR, "../models/face_label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

with open(os.path.join(BASE_DIR, "../models/face_classifier_model.pkl"), "rb") as f:
    classifier = pickle.load(f)

print("Model loaded successfully.")


# =========================
# Preprocessing & Augmentasi
# =========================
def preprocess_face(image_bytes):
    """Decode gambar, deteksi wajah, dan resize ke 160x160."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Gambar tidak valid atau gagal di-decode.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = os.path.join(BASE_DIR, "../models/haarcascade_frontalface_default.xml")

        if not os.path.exists(cascade_path):
            raise FileNotFoundError("File Haar Cascade tidak ditemukan.")

        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None, None

        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w]
            face = cv2.resize(face, (160, 160))
            return face, img

        return None, None

    except Exception as e:
        raise RuntimeError(f"Gagal preprocess wajah: {str(e)}")


def apply_random_augmentation(image):
    """Lakukan augmentasi random: brightness, rotasi, flip, noise."""
    # Random brightness
    alpha = random.uniform(0.7, 1.3)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=random.randint(-10, 10))

    # Random rotation
    angle = random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))

    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Tambah sedikit noise
    noise = np.random.normal(0, 3, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)

    return image


# =========================
# Face Recognition
# =========================
def recognize_face(image_bytes):
    face, _ = preprocess_face(image_bytes)

    if face is None:
        return {"status": "no_face", "message": "Tidak ada wajah terdeteksi."}

    # Prediksi dari wajah asli
    embedding_main = embedder.embeddings([face])[0]
    pred = classifier.predict([embedding_main])[0]
    prob = classifier.predict_proba([embedding_main])[0].max()
    label = label_encoder.inverse_transform([pred])[0]

    THRESHOLD = 0.7  # ganti sesuai kebutuhan
    if prob < THRESHOLD:
        return {
            "status": "failed",
            "message": "Wajah tidak dikenali.",
            "confidence": round(prob, 4),
            "waktu": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        }

    # Jika lolos threshold → buat embedding tambahan untuk log
    embeddings = []
    for i in range(7):
        augmented_face = apply_random_augmentation(face.copy())
        embedding = embedder.embeddings([augmented_face])[0].tolist()
        embeddings.append(embedding)

    for emb in embeddings:
        emb_noisy = [v + random.uniform(-0.01, 0.01) for v in emb]
        save_prediction_log(label, label, label, emb_noisy)

    return {
        "status": "success",
        "label": label,
        "confidence": round(prob, 4),
        "waktu": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    }


# =========================
# Log Management
# =========================
def get_log_data():
    """Baca log dari CSV."""
    log_file = "log/log.csv"
    data = []

    if not os.path.exists(log_file):
        return data

    with open(log_file, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    return data


def save_log_entry(entry):
    """Simpan satu entry ke file log CSV."""
    os.makedirs("log", exist_ok=True)
    log_file = "log/log.csv"

    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if os.stat(log_file).st_size == 0:
            writer.writerow(["nama", "waktu", "akurasi", "status"])

        writer.writerow([
            entry.get("nama", "Tidak diketahui"),
            entry.get("waktu", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            entry.get("akurasi", "0%"),
            entry.get("status", "unknown"),
        ])

    return {"status": "success", "message": "Log saved"}


def determine_session(label):
    """Fungsi dummy untuk menentukan sesi."""
    return "Sesi Otentikasi"


def delete_log_entry(timestamp: str):
    """Hapus data log berdasarkan timestamp."""
    log_file = "log/log.csv"
    if not os.path.exists(log_file):
        return {"status": "error", "message": "Log file tidak ditemukan"}

    rows = []
    deleted = False

    with open(log_file, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["waktu"] == timestamp:
                deleted = True
                continue
            rows.append(row)

    if not deleted:
        return {"status": "error", "message": "Data tidak ditemukan"}

    with open(log_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["nama", "waktu", "akurasi", "status"])
        writer.writeheader()
        writer.writerows(rows)

    return {"status": "success", "message": "Log berhasil dihapus"}


# =========================
# FastAPI Endpoint
# =========================
@auth_router.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = recognize_face(image_bytes)

        if result.get("status") == "success":
            nama_user = result.get("label")
            confidence = result.get("confidence", 1.0)
            run_evaluation(nama_user, confidence)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
