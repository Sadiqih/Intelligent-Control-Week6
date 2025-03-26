import cv2
import numpy as np
import matplotlib.pyplot as plt  # Tambahkan ini

def canny_edge_detection(image_path):
    """Mendeteksi tepi menggunakan metode Canny Edge Detection"""

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img_blur, 50, 150)

    # Gunakan Matplotlib untuk menampilkan gambar
    plt.figure(figsize=(8, 6))
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edge Detection")
    plt.axis("off")
    plt.show()

    # Simpan hasilnya
    cv2.imwrite("canny_result.jpg", edges)

    return "canny_result.jpg"

# Contoh penggunaan
canny_edge_detection("D:/Phyton/Intelligent-Control-Week6/istockphoto-135096747-1024x1024.jpg")
