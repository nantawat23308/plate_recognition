# Bounding Box Conversion Guide

## Overview
When working with nested detections (detecting objects within cropped regions), you need to convert bounding box coordinates back to the original image space. This guide explains how to do it.

## Problem Scenario

Your plate recognition system has this workflow:
1. **Original Image** (e.g., 1920x1080)
2. **Detect Cars** → Get car bounding boxes in original coordinates
3. **Crop Car Region** → Extract car region (e.g., 400x300)
4. **Detect Plates** → Get plate bounding boxes **in the cropped car region**
5. **Convert to Original Space** → Map plate coordinates back to original image

## Solution: Using Offsets

### Key Concept
When you crop a region starting at position `(x_offset, y_offset)`, any coordinates in that crop need to be adjusted by adding the offset:

```
Original Coordinate = Cropped Coordinate + Offset
```

### Example

**Original Image: 1920x1080**

Car detected at: `(100, 50, 500, 400)`
- Top-left: (100, 50)
- Bottom-right: (500, 400)

**Crop car region:**
```python
crop_car = frame[50:400, 100:500]  # height range, width range
offset = (100, 50)  # x_offset, y_offset
```

**Detect plate in crop:**
Plate detected at: `(50, 30, 150, 70)` in the cropped image

**Convert back to original coordinates:**
```python
x1_original = 50 + 100 = 150
y1_original = 30 + 50 = 80
x2_original = 150 + 100 = 250
y2_original = 70 + 50 = 120

bbox_original = (150, 80, 250, 120)
```

## Implementation in Your Code

### Updated `PlateDetector.detect_plate()`

```python
def detect_plate(self, frame, offset=(0, 0)) -> list:
    """
    Detect plates in frame and return cropped plates with original coordinates.
    
    Args:
        frame: The input image (can be full image or cropped image)
        offset: Tuple (x_offset, y_offset) to convert coordinates to original image space
               Use this when frame is a cropped region of a larger image
    
    Returns:
        List of dicts containing:
        - 'crop': cropped plate image
        - 'bbox': coordinates in original image
        - 'bbox_in_crop': coordinates in cropped car image
    """
    results = self.model.predict(source=frame)
    detected_plates = []

    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Convert to original image coordinates
            x1_orig = x1 + offset[0]
            y1_orig = y1 + offset[1]
            x2_orig = x2 + offset[0]
            y2_orig = y2 + offset[1]
            
            # Crop the plate
            h, w, _ = frame.shape
            crop = frame[max(0, y1 - 5):min(h, y2 + 5), max(0, x1 - 5):min(w, x2 + 5)]
            
            detected_plates.append({
                'crop': crop,
                'bbox': (x1_orig, y1_orig, x2_orig, y2_orig),
                'bbox_in_crop': (x1, y1, x2, y2)
            })
    return detected_plates
```

### Updated `main.py` Processing

```python
def process_image(self, image_path):
    frame = self.read_image(image_path)
    car_detected = self.car_detector.detect_car(frame)
    
    for car in car_detected:
        x1_car, y1_car = car[0]  # top-left corner of car
        x2_car, y2_car = car[2]  # bottom-right corner of car
        
        crop_car = self.crop_plate(frame, car)
        
        # Pass offset to convert plate coordinates back to original image
        plate_detected = self.plate_detector.detect_plate(
            crop_car, 
            offset=(x1_car, y1_car)
        )
        
        for plate_info in plate_detected:
            plate_crop = plate_info['crop']
            bbox_original = plate_info['bbox']  # Coordinates in original image!
            
            text, _ = self.recognizer.predict(plate_crop)
            print(f"Recognized Plate Text: {text}")
            print(f"Plate Bounding Box (original image): {bbox_original}")
```

## Key Points to Remember

1. **Offset = Crop Start Position**: When cropping with `frame[y1:y2, x1:x2]`, the offset is `(x1, y1)`

2. **Always Preserve Cropped Coordinates**: Keep track of both:
   - Coordinates in the cropped image (for drawing on the crop)
   - Coordinates in the original image (for final visualization/output)

3. **Multiple Levels of Nesting**: If you have 3+ levels, add offsets at each level:
   ```python
   offset_level2 = (car_x, car_y)
   offset_level3 = (offset_level2[0] + plate_x, offset_level2[1] + plate_y)
   ```

4. **Visualization**: Draw bounding boxes on the original image using original coordinates:
   ```python
   x1, y1, x2, y2 = bbox_original
   cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
   ```

## Testing Your Implementation

```python
import cv2

# Load image
image = cv2.imread("image.jpg")

# Detect cars
cars = car_detector.detect_car(image)

for car in cars:
    # Get car coordinates
    x1_car, y1_car = car[0]
    x2_car, y2_car = car[2]
    
    # Crop
    crop_car = image[y1_car:y2_car, x1_car:x2_car]
    
    # Detect plates with offset
    plates = plate_detector.detect_plate(crop_car, offset=(x1_car, y1_car))
    
    for plate_info in plates:
        bbox = plate_info['bbox']
        # This should work correctly on the original image
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

# Show result
cv2.imshow("Result", image)
cv2.waitKey(0)
```

## Visual Example

```
Original Image (1920x1080):
┌─────────────────────────────────────┐
│  Car Region (100,50 to 500,400)     │
│  ┌───────────────────────────────┐  │
│  │  Plate (150,80 to 250,120)    │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │                         │  │  │
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘

In Cropped Car Image (400x350):
offset = (100, 50)
┌───────────────────────────────┐
│ Plate (50,30 to 150,70)       │
│ ┌─────────────────────────┐   │
│ │                         │   │
│ └─────────────────────────┘   │
└───────────────────────────────┘

Conversion:
x1_orig = 50 + 100 = 150 ✓
y1_orig = 30 + 50 = 80 ✓
```

---

Your code has been updated with this implementation. Happy plate recognition! 🚗📸

