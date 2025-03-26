import cv2
import numpy as np
import torch
from ultralytics import YOLO  # Pastikan ultralytics sudah terinstal

# Pastikan file model tersedia di lokasi yang benar
MODEL_PATH = "model-segmentation-rel.pt"  # Model YOLO yang telah dilatih untuk mendeteksi rel
IMAGE_PATH = "Jalan-Rel-Kereta-Api.jpg"  # Ganti dengan gambar yang sesuai

# Memuat model YOLOv8
try:
    model = YOLO(MODEL_PATH)  # Gunakan YOLO jika model berasal dari Ultralytics
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# Daftar label yang dianggap sebagai "rel kereta"
REL_CLASSES = ["rail", "train track", "rel kereta"]  # Sesuaikan dengan label dataset Anda

# Fungsi utama untuk instance segmentation pada rel kereta
def segment_rels():
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print("Error: Gambar tidak ditemukan atau gagal dimuat!")
        return

    # Konversi frame ke RGB untuk YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prediksi menggunakan model YOLO
    results = model(frame_rgb)  # Lakukan prediksi
    predictions = results[0]  # Ambil hasil prediksi pertama

    # Debugging - Cek apakah ada deteksi objek
    print("Deteksi objek:", predictions.boxes.cls if predictions.boxes else "Tidak ada")
    print("Mask segmentasi:", predictions.masks if predictions.masks else "Tidak ada")

    # Buat overlay transparan untuk segmentasi warna
    overlay = frame.copy()

    # Mask kosong untuk rel
    mask_model = np.zeros_like(frame[:, :, 0], dtype=np.uint8)

    # Cek apakah model mendukung segmentasi dan mendeteksi objek
    if predictions.masks is not None and predictions.boxes is not None:
        for i, mask in enumerate(predictions.masks.data):  # Menggunakan mask biner
            label_idx = int(predictions.boxes.cls[i])
            label = model.names[label_idx] if label_idx in model.names else "Unknown"

            # Hanya proses objek yang termasuk dalam kategori rel
            if label.lower() in REL_CLASSES:
                mask = mask.cpu().numpy().astype(np.uint8) * 255  # Konversi ke uint8
                mask_model = cv2.bitwise_or(mask_model, mask)  # Gabungkan mask
                
                # Tambahkan warna biru pada rel
                overlay[mask == 255] = (255, 0, 0)  # Biru

                # Tambahkan bounding box & label
                x1, y1, x2, y2 = map(int, predictions.boxes.xyxy[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Gabungkan overlay dengan transparansi 50%
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Tampilkan gambar dengan ukuran yang sesuai
    h, w = frame.shape[:2]
    scale_factor = 800 / max(h, w)  # Skala agar gambar lebih besar di layar
    frame_resized = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))

    cv2.imshow('Instance Segmentation - Rel Kereta', frame_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Jalankan program
segment_rels()
