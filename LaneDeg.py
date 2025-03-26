from ultralytics import YOLO

import cv2



# Load model YOLOv8 Instance Segmentation

model = YOLO("yolov8n-seg.pt")



def detect_rail_lane(image_path):

    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""

    results = model(image_path, show=True)

    results[0].save("lane_detection_result.jpg")



# Contoh penggunaan

detect_rail_lane("678f354157750-rel-kereta-api-terendam-banjir-di-grobogan_jateng.jpg")

from ultralytics import YOLO

import cv2



# Load model YOLOv8 Instance Segmentation

model = YOLO("yolov8n-seg.pt")



def detect_rail_lane(image_path):

    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""

    results = model(image_path, show=True)

    results[0].save("lane_detection_result.jpg")



# Contoh penggunaan

detect_rail_lane("678f354157750-rel-kereta-api-terendam-banjir-di-grobogan_jateng.jpg")