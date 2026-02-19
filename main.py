from ultralytics import YOLO
from roboflow import Roboflow
import os
from dotenv import load_dotenv
from src.annotate import extract_corners
from src.detect_car import CarDetector
from src.detect_plate import PlateDetector
from src.recognizer import PaddleOCRPipeline
import cv2
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

    def process_image(self, image_path):
        frame = self.read_image(image_path)
        car_detected = self.car_detector.detect_car(frame)
        car_image = []
        for car in car_detected:
            print(f"Detected car with corners: {car}")
            crop_car = self.crop_plate(frame, car)
            car_image.append(crop_car)

        for car in car_image:
            plate_detected = self.plate_detector.detect_plate(car)
            for plate in plate_detected:
                text = self.recognizer.predict(plate)
                print(f"Recognized Plate Text: {text}")



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


if __name__ == "__main__":
    pass