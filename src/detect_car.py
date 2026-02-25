from ultralytics import YOLO
from roboflow import Roboflow
import os
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import cv2

class CarDetector:
    def __init__(self, model_path="yolo26n.pt"):
        self.model = YOLO(model_path)
        self.car_classes = [2, 3, 5, 7]  # COCO classes for car, motorcycle, bus, truck

    def detect_car(self, input_data) -> list:
        """
        Detect cars in image.

        Args:
            input_data: Can be either image path (str) or numpy array (frame)

        Returns:
            List of car corner coordinates
        """
        results = self.model.predict(source=input_data, classes=self.car_classes)
        detected_cars = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                corners = self.extract_corners(box)
                detected_cars.append(corners)
                print(f"Detected car with corners: {corners}")
        return detected_cars

    @staticmethod
    def extract_corners(box_tensor):
        x1, y1, x2, y2 = box_tensor.xyxy[0].astype(int).tolist()
        top_left = (x1, y1)
        top_right = (x2, y1)
        bottom_right = (x2, y2)
        bottom_left = (x1, y2)
        return [top_left, top_right, bottom_right, bottom_left]


    def plot_results(self, image_path):
        image = cv2.imread(image_path)
        detect_cars = self.detect_car(image)
        for car in detect_cars:
            print(f"Corner: {car}")
            top_left, top_right, bottom_right, bottom_left = car
            print(f"Top Left: {top_left}, Top Right: {top_right}, Bottom Right: {bottom_right}, Bottom Left: {bottom_left}")
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 10)
        plt.imsave("car_detection_result.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    car_detector = CarDetector()
    car_detector.plot_results("/home/nantawat/Desktop/my_project/plate_recognition/dataset/l711n5sj8se31.jpg")
