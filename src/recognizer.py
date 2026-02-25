from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont # Needed for Thai text
import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_thai_text(img, text, position, color=(255, 0, 0), font_size=30):
    """
    OpenCV cannot draw Thai text. We use PIL to draw it, then convert back to OpenCV.
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Load your font (Ensure 'tahoma.ttf' is in your folder or give full path)
    # If on Windows, you can try: "C:/Windows/Fonts/tahoma.ttf"
    try:
        font_path = "/usr/share/fonts/truetype/tlwg/Kinnari.ttf" # tahoma.ttf

        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()  # Fallback if font not found
        print("Warning: Thai font not found. Text might look wrong.")

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class PaddleOCRPipeline:
    def __init__(self, ocr_lang="th"):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            lang=ocr_lang
        )

    def predict(self, input) -> tuple[str, list]:
        results = self.ocr.predict(input)
        detected_texts = ""
        rectangle = []
        for result in results:
            print(result.keys())
            rectangle = result.get("rec_boxes")
            rec_text = (result.get("rec_texts"))
            conf = result.get("rec_scores")
            print(f"Detected: {rec_text} with confidence {conf}")
            detected_texts += " ".join(rec_text) + " "
        return detected_texts.strip(), rectangle


if __name__ == "__main__":
    ocr_pipeline = PaddleOCRPipeline()
    image_path = "/home/nantawat/Desktop/my_project/plate_recognition/dataset/img_3.png"
    detected_text, box = ocr_pipeline.predict(image_path)
    print(f"Final Detected Text: {detected_text}")
    print(f"Bounding Box: {box}")