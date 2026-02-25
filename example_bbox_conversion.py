"""
Example: How to Draw Bounding Boxes on Original Image

This script demonstrates how to:
1. Detect cars in original image
2. Crop car regions
3. Detect plates in car regions
4. Draw all bounding boxes on original image with correct coordinates
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.detect_car import CarDetector
from src.detect_plate import PlateDetector
from src.recognizer import PaddleOCRPipeline


def draw_bboxes_on_image(image, cars, plates_dict):
    """
    Draw car and plate bounding boxes on the image.

    Args:
        image: Original image (numpy array)
        cars: List of car corners
        plates_dict: Dict with car_index -> list of plate bboxes in original coords
    """
    # Draw car bounding boxes (green)
    for car_idx, car in enumerate(cars):
        x1, y1 = car[0]
        x2, y2 = car[2]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image, f"Car {car_idx}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw plate bounding boxes (blue)
    if plates_dict:
        for car_idx, plates in plates_dict.items():
            for plate_idx, bbox in enumerate(plates):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f"Plate {plate_idx}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    return image


def main_example():
    """
    Complete example showing coordinate conversion.
    """
    # Initialize detectors
    car_detector = CarDetector("yolo26n.pt")
    plate_detector = PlateDetector("yolo26_coco128_plate/best.pt")
    recognizer = PaddleOCRPipeline()

    # Load image
    image_path = "dataset/img_1.png"
    original_image = cv2.imread(image_path)
    working_image = original_image.copy()

    print(f"Original image shape: {original_image.shape}")

    # ===== STEP 1: Detect Cars =====
    cars = car_detector.detect_car(original_image)
    print(f"\nDetected {len(cars)} car(s)")

    plates_dict = {}  # Store plates with their original coordinates

    # ===== STEP 2: For each car, detect plates =====
    for car_idx, car in enumerate(cars):
        print(f"\n--- Processing Car {car_idx} ---")

        # Get car coordinates
        x1_car, y1_car = car[0]  # Top-left
        x2_car, y2_car = car[2]  # Bottom-right

        print(f"Car coordinates (original): ({x1_car}, {y1_car}) to ({x2_car}, {y2_car})")

        # Crop the car region
        crop_car = original_image[y1_car:y2_car, x1_car:x2_car]
        print(f"Car crop shape: {crop_car.shape}")

        # ===== STEP 3: Detect plates in the cropped car =====
        # IMPORTANT: Pass the offset so coordinates are converted back!
        plate_results = plate_detector.detect_plate(crop_car, offset=(x1_car, y1_car))

        print(f"Detected {len(plate_results)} plate(s) in this car")

        plates_dict[car_idx] = []

        # ===== STEP 4: Process each plate =====
        for plate_idx, plate_info in enumerate(plate_results):
            plate_crop = plate_info['crop']
            bbox_original = plate_info['bbox']  # ← This is in ORIGINAL image coords!
            bbox_in_crop = plate_info['bbox_in_crop']

            print(f"\n  Plate {plate_idx}:")
            print(f"    Original coordinates: {bbox_original}")
            print(f"    Crop coordinates: {bbox_in_crop}")

            # Recognize text
            text, _ = recognizer.predict(plate_crop)
            print(f"    Recognized text: {text}")

            # Store the original coordinates
            plates_dict[car_idx].append(bbox_original)

    # ===== STEP 5: Draw all bounding boxes on original image =====
    result_image = draw_bboxes_on_image(working_image, cars, plates_dict)

    # Save and display
    # cv2.imwrite("result_with_bboxes.jpg", result_image)
    # cv2.imshow("Result with Bounding Boxes", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(result_image)
    plt.axis("off")
    plt.show()
    print("\n✓ Result saved as 'result_with_bboxes.jpg'")


if __name__ == "__main__":
    main_example()

