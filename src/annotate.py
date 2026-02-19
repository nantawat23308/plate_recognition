from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_results(results, image_path, model):
    img = cv2.imread(image_path)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            print(f"Box: ({x1}, {y1}, {x2}, {y2}), Confidence: {conf}, Class: {cls}")
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            class_name = model.names[int(cls)]
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label,
                        (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def extract_corners(box):
    """
    Extract corners from the bounding box.
    YOLO gives us (x1, y1, x2, y2) which are the top-left and bottom-right corners.
    We can derive the other two corners from these.
    """
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    top_left = (x1, y1)
    top_right = (x2, y1)
    bottom_right = (x2, y2)
    bottom_left = (x1, y2)
    return [top_left, top_right, bottom_right, bottom_left]

def align_plate(image, corners):
    """
    Standard approach to fix slanted plates using 4 corners.
    corners: list of [top-left, top-right, bottom-right, bottom-left]
    """
    # 1. Destination points (The ideal flat rectangle)
    # Standard License Plate ratio is roughly 4:1 or 2:1 depending on country
    width, height = 200, 50
    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # 2. Source points (The corners detected by YOLO-Pose)
    src_pts = np.float32(corners)

    # 3. Get the transformation matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 4. Warp (Flatten) the image
    warped_plate = cv2.warpPerspective(image, matrix, (width, height))

    return warped_plate




def main():
    # Load pretrained model
    model = YOLO(
        "/home/nantawat/Desktop/my_project/plate_recognition/yolo26_coco128_plate/best.pt")  # or your custom model

    image_folder = "dataset/images/train"
    label_folder = "dataset/labels/train"

    os.makedirs(label_folder, exist_ok=True)

    my_test = "/home/nantawat/Desktop/my_project/plate_recognition/img.png"

    results = model.predict(source=my_test, save_txt=True, save_conf=False)
    img = cv2.imread(my_test)
    # show_results(results, my_test, model)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            corners = extract_corners(box)
            aligned_plate = align_plate(img, corners)
            plt.imshow(cv2.cvtColor(aligned_plate, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

if __name__ == '__main__':
    main()