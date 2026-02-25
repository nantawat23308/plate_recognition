import numpy as np
from ultralytics import YOLO
from roboflow import Roboflow
import os
from dotenv import load_dotenv
from src.annotate import extract_corners
from src.detect_car import CarDetector
from src.detect_plate import PlateDetector
from src.recognizer import PaddleOCRPipeline, draw_thai_text
import cv2
import matplotlib.pyplot as plt
load_dotenv()

class CarPlateDetector:
    def __init__(self, model_path="yolo26_coco128_plate/best.pt"):
        self.model = YOLO(model_path)
        self.car_detector = CarDetector(
                "/home/nantawat/Desktop/my_project/plate_recognition/yolo26n.pt"
        )
        self.plate_detector = PlateDetector(
                "/home/nantawat/Desktop/my_project/plate_recognition/yolo26_coco128_plate/best.pt"
        )
        self.recognizer = PaddleOCRPipeline()

    def read_image(self, image_path):
        return cv2.imread(image_path)


    def crop_plate(self, frame, corners):
        x1, y1 = corners[0]  # top-left
        x2, y2 = corners[2]  # bottom-right
        h, w, _ = frame.shape
        crop = frame[max(0, y1 - 5):min(h, y2 + 5), max(0, x1 - 5):min(w, x2 + 5)]
        return crop

    def process_image(self, image_path) -> np.ndarray:
        frame = self.read_image(image_path)
        width, height = frame.shape[1], frame.shape[0]
        car_detected = self.car_detector.detect_car(frame)

        for car in car_detected:
            print(f"Detected car with corners: {car}")
            x1_car, y1_car = car[0]  # top-left corner of car
            x2_car, y2_car = car[2]  # bottom-right corner of car

            crop_car = self.crop_plate(frame, car)

            # Detect plates in the cropped car image
            # Pass offset to convert plate coordinates back to original image space
            plate_detected = self.plate_detector.detect_plate(crop_car, offset=(x1_car, y1_car))

            for plate_info in plate_detected:
                plate_crop = plate_info['crop']
                bbox_original = plate_info['bbox']  # Coordinates in original image
                bbox_in_crop = plate_info['bbox_in_crop']  # Coordinates in cropped car image

                text, ocr_bbox = self.recognizer.predict(plate_crop)
                print(f"Recognized Plate Text: {text}")
                print(f"Plate Bounding Box (original image): {bbox_original}")
                print(f"Plate Bounding Box (in crop): {bbox_in_crop}")
                cv2.rectangle(frame, (bbox_original[0], bbox_original[1]), (bbox_original[2], bbox_original[3]), (0, 255, 0), 2)
                frame = draw_thai_text(frame, text, (bbox_original[0], bbox_original[1] - 40), color=(255, 0, 0))
                # cv2.putText(frame, text, (bbox_original[0], bbox_original[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # plt.imsave("result.png", frame)
        plt.imshow(frame)
        plt.axis("off")
        plt.show()

        return frame


def detect_car(image_path):
    car_class = [2, 3, 5, 7]  # COCO classes for car, motorcycle, bus, truck
    model = YOLO("yolo26n.pt")
    results = model.predict(source=image_path, classes=car_class)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            corners = extract_corners(box)
            print(f"Detected car with corners: {corners}")


def detect_plate(frame):
    model = YOLO("")

def main():
    image_path = "/home/nantawat/Desktop/my_project/plate_recognition/dataset/img_1.png"
    car_plate_detector = CarPlateDetector()
    car_plate_detector.process_image(image_path)

if __name__ == "__main__":
    main()