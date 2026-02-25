# Bounding Box Conversion - Cheat Sheet

## The Formula (Remember This!)

```
ORIGINAL = CROP + OFFSET
```

Where:
- **CROP** = coordinates in the cropped/sub-image
- **OFFSET** = position where crop started in original image
- **ORIGINAL** = coordinates in the full original image

## OpenCV Basics

```python
# Reading and basic operations
image = cv2.imread("image.jpg")          # BGR format
height, width = image.shape[:2]          # Get dimensions

# Cropping (IMPORTANT: y comes first!)
crop = image[y1:y2, x1:x2]              # ✓ Correct: [row, column]
crop = image[x1:x2, y1:y2]              # ✗ Wrong!

# Drawing (IMPORTANT: x comes first!)
cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)  # ✓ Correct
cv2.circle(image, (x, y), radius, color, thickness)         # ✓ Correct
```

## Your Implementation - Quick Copy-Paste

### For Detecting in Cropped Car Image

```python
# Get car position
x1_car, y1_car = car[0]  # top-left
x2_car, y2_car = car[2]  # bottom-right

# Crop the car
crop_car = frame[y1_car:y2_car, x1_car:x2_car]

# Detect plates with offset (THE KEY LINE!)
plates = plate_detector.detect_plate(crop_car, offset=(x1_car, y1_car))

# Use the result (already in original coordinates!)
for plate_info in plates:
    bbox_original = plate_info['bbox']
    cv2.rectangle(frame, (bbox_original[0], bbox_original[1]),
                  (bbox_original[2], bbox_original[3]), (0, 255, 0), 2)
```

### For Multiple Levels of Crops

```python
# Level 1: Original → Car
offset_car = (car_x, car_y)
crop_car = frame[car_y:car_y2, car_x:car_x2]

# Level 2: Car → Plate
offset_plate_in_car = (plate_x, plate_y)
offset_plate_original = (
    offset_car[0] + offset_plate_in_car[0],
    offset_car[1] + offset_plate_in_car[1]
)

# Or let the code do it:
plates = plate_detector.detect_plate(crop_car, offset=offset_car)
# bbox is already converted!
```

## Common Patterns

### Pattern 1: Just Get Coordinates
```python
# Get plate coordinates in original image
plate_info = plate_detected[0]
x1, y1, x2, y2 = plate_info['bbox']
print(f"Plate at: ({x1}, {y1}) to ({x2}, {y2})")
```

### Pattern 2: Draw All Detections
```python
# Draw cars
for car in cars:
    cv2.rectangle(image, car[0], car[2], (0, 255, 0), 2)

# Draw plates
for car_idx, car in enumerate(cars):
    crop_car = image[car[0][1]:car[2][1], car[0][0]:car[2][0]]
    plates = plate_detector.detect_plate(crop_car, offset=car[0])
    
    for plate in plates:
        bbox = plate['bbox']
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                     (255, 0, 0), 1)

cv2.imshow("Result", image)
```

### Pattern 3: Crop and Save Individual Plates
```python
for plate_info in plates:
    plate_crop = plate_info['crop']  # Already cropped!
    bbox = plate_info['bbox']
    
    # Save
    cv2.imwrite(f"plate_{bbox[0]}.jpg", plate_crop)
    
    # Recognize text
    text, _ = recognizer.predict(plate_crop)
    print(f"Text: {text}, Position: {bbox}")
```

## Coordinate Spaces Cheat Sheet

```
SPACE 1: Original Image
├─ Size: Full image (e.g., 1920x1080)
├─ Used for: Final drawing, output coordinates
└─ Get from: plate_info['bbox']

SPACE 2: Car Crop
├─ Size: Cropped car region (e.g., 400x350)
├─ Used for: Detecting plates in the crop
├─ Offset from original: (car_x1, car_y1)
└─ Use when: Working with cropped image

SPACE 3: Plate Crop
├─ Size: Tiny plate image
├─ Used for: OCR recognition
├─ Coordinates: Small values (50, 30, 150, 70)
└─ Get from: plate_info['crop']
```

## Offset Calculation Quick Guide

```
Given:  image[y1:y2, x1:x2] ← Crop bounds
        
Then:   offset = (x1, y1)   ← The offset

Not:    offset = (y1, x1)   ← Wrong!
        offset = (x2, y2)   ← Wrong!
```

## Debugging Checklist

When bounding boxes are in the wrong position:

- [ ] Did I pass the `offset` parameter?
  ```python
  plates = plate_detector.detect_plate(crop_car, offset=(car_x, car_y))
  # YES? Continue...
  ```

- [ ] Am I using the right bbox?
  ```python
  bbox = plate_info['bbox']      # ✓ For original image
  bbox = plate_info['bbox_in_crop']  # ✗ For cropped image only
  ```

- [ ] Is my offset in the right order?
  ```python
  offset = (x_value, y_value)    # ✓ Correct: x first
  offset = (y_value, x_value)    # ✗ Wrong!
  ```

- [ ] Am I cropping with the right syntax?
  ```python
  crop = image[y1:y2, x1:x2]     # ✓ Correct: y first
  crop = image[x1:x2, y1:y2]     # ✗ Wrong!
  ```

- [ ] Is my offset matching the crop start?
  ```python
  crop = image[100:500, 50:300]      # Top-left at (50, 100)
  offset = (50, 100)                  # ✓ Correct!
  # NOT: offset = (100, 50)           # ✗ Wrong!
  ```

## One-Liner Examples

```python
# Convert crop coords to original coords
original_x1 = crop_x1 + offset_x
original_y1 = crop_y1 + offset_y

# Check if point is in crop
if 0 <= point_x - offset_x < crop_width and \
   0 <= point_y - offset_y < crop_height:
    print("Point is in crop")

# Convert multiple boxes
boxes_original = [(x + ox, y + oy, x2 + ox, y2 + oy) 
                  for x, y, x2, y2 in boxes_crop]

# Get crop offset from bounding box
x1, y1, x2, y2 = bbox
crop_offset = (x1, y1)
crop = image[y1:y2, x1:x2]
```

## Test It!

```python
# Minimal test
import cv2
from src.detect_car import CarDetector
from src.detect_plate import PlateDetector

car_det = CarDetector("yolo26n.pt")
plate_det = PlateDetector("yolo26_coco128_plate/best.pt")

img = cv2.imread("test.jpg")
cars = car_det.detect_car(img)

for car in cars:
    crop = img[car[0][1]:car[2][1], car[0][0]:car[2][0]]
    plates = plate_det.detect_plate(crop, offset=car[0])
    
    print(f"Found {len(plates)} plates")
    for p in plates:
        print(f"  Bbox: {p['bbox']}")
        cv2.rectangle(img, p['bbox'][:2], p['bbox'][2:], (0,255,0), 2)

cv2.imshow("Test", img)
cv2.waitKey(0)
```

---

**Remember:** When in doubt, add the offset! 📍

