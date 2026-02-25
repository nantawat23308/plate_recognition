from src.annotate import extract_corners, align_plate, show_results
from src.recognizer import draw_thai_text
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from paddleocr import PaddleOCR


class PlateDetector:
    def __init__(self, model_path="yolo26_coco128_plate/best.pt"):
        self.model = YOLO(model_path)

    def detect_plate(self, frame, offset=(0, 0)) -> list:
        """
        Detect plates in frame and return cropped plates with original coordinates.

        Args:
            frame: The input image (can be full image or cropped image)
            offset: Tuple (x_offset, y_offset) to convert coordinates to original image space
                   Use this when frame is a cropped region of a larger image

        Returns:
            List of dicts containing 'crop' (cropped plate image) and 'bbox' (coords in original image)
        """
        results = self.model.predict(source=frame)
        detected_plates = []

        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                x1, y1, x2, y2 = box

                # Convert to original image coordinates if offset is provided
                x1_orig = x1 + offset[0]
                y1_orig = y1 + offset[1]
                x2_orig = x2 + offset[0]
                y2_orig = y2 + offset[1]

                # A. CROP THE PLATE
                h, w, _ = frame.shape
                crop = frame[max(0, y1 - 5):min(h, y2 + 5), max(0, x1 - 5):min(w, x2 + 5)]

                detected_plates.append({
                    'crop': crop,
                    'bbox': (x1_orig, y1_orig, x2_orig, y2_orig),
                    'bbox_in_crop': (x1, y1, x2, y2)
                })
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