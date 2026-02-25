from src.annotate import extract_corners, align_plate, show_results
from src.recognizer import draw_thai_text
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
    lang="th"
)

if __name__ == '__main__':
    model = YOLO(
            "/home/nantawat/Desktop/my_project/plate_recognition/yolo26_coco128_plate/best.pt")
    frame = cv2.imread("/home/nantawat/Desktop/my_project/plate_recognition/dataset/img_1.png")
    results = model.predict(source=frame)

    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            x1, y1, x2, y2 = box
            # A. CROP THE PLATE
            # Add a small margin (padding) to ensure we don't cut the edge of the text
            h, w, _ = frame.shape
            crop = frame[max(0, y1 - 5):min(h, y2 + 5), max(0, x1 - 5):min(w, x2 + 5)]
            ocr_result = ocr.predict(crop)
            detected_text = ""

            # Paddle returns a list of lines. Thai plates often have 2 lines.
            for result in ocr_result:
                rec_text = (result.get("rec_texts"))
                conf = (result.get("rec_scores"))

                # if conf > 0.5:
                #     detected_text += rec_text + " "
                detected_text += " ".join(rec_text)

            print(f"Detected: {detected_text}")

            # C. DRAW RESULT
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Use our special function to draw Thai text
            frame = draw_thai_text(frame, detected_text, (x1, y1 - 40))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
