from src.annotate import extract_corners, align_plate, show_results
from src.recognizer import draw_thai_text
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from paddleocr import PaddleOCR


class PlateDetector:
    def __init__(self, model_path="yolo26_coco128_plate/best.pt"):
        self.model = YOLO(model_path)

    def detect_plate(self, frame) -> list:
        results = self.model.predict(source=frame)
        detected_plates = []

        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                x1, y1, x2, y2 = box
                # A. CROP THE PLATE
                h, w, _ = frame.shape
                crop = frame[max(0, y1 - 5):min(h, y2 + 5), max(0, x1 - 5):min(w, x2 + 5)]
                detected_plates.append(crop)
        return detected_plates

if __name__ == '__main__':
    plate_detector = PlateDetector("/home/nantawat/Desktop/my_project/plate_recognition/yolo26_coco128_plate/best.pt")
    frame = cv2.imread("/home/nantawat/Desktop/my_project/plate_recognition/dataset/l711n5sj8se31.jpg")
    detected_plates = plate_detector.detect_plate(frame)
    for idx, plate in enumerate(detected_plates):
        plt.subplot(1, len(detected_plates), idx + 1)
        plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        plt.axis("off")
    plt.show()