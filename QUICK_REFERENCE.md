# Quick Reference: Bounding Box Coordinate Conversion

## The Simple Formula

```
Original Coordinate = Cropped Coordinate + Offset
```

Where:
- **Cropped Coordinate**: Position in the cropped/sub-image
- **Offset**: Position where the crop started in the parent image
- **Original Coordinate**: Position in the full original image

## Your Use Case

### Pattern 1: Single Crop Level
```python
# Detect something and crop
x1, y1, x2, y2 = bbox_in_image
crop = image[y1:y2, x1:x2]
offset = (x1, y1)

# Detect in crop
x1_crop, y1_crop, x2_crop, y2_crop = detect(crop)

# Convert back
x1_orig = x1_crop + offset[0]
y1_orig = y1_crop + offset[1]
```

### Pattern 2: Multiple Crop Levels (YOUR CASE)
```python
# Level 1: Original Image → Car Crop
image_shape = (1920, 1080)
car_bbox = (100, 50, 500, 400)  # x1, y1, x2, y2
car_crop = image[50:400, 100:500]
car_offset = (100, 50)

# Level 2: Car Crop → Plate Crop
plate_in_car = (50, 30, 150, 70)  # detected in car_crop
plate_crop = car_crop[30:70, 50:150]
plate_offset_in_car = (50, 30)

# Convert to original image
plate_offset_in_image = (
    car_offset[0] + plate_offset_in_car[0],  # 100 + 50 = 150
    car_offset[1] + plate_offset_in_car[1]   # 50 + 30 = 80
)
# OR if using full bbox:
x1_orig = plate_in_car[0] + car_offset[0]  # 50 + 100 = 150
y1_orig = plate_in_car[1] + car_offset[1]  # 30 + 50 = 80
```

## Code Changes Made

### 1. `src/detect_plate.py` - Updated `detect_plate()`
**Now accepts an `offset` parameter to automatically convert coordinates:**

```python
def detect_plate(self, frame, offset=(0, 0)) -> list:
    # Returns list of dicts:
    # {
    #     'crop': cropped_plate_image,
    #     'bbox': (x1, y1, x2, y2) in ORIGINAL image,
    #     'bbox_in_crop': (x1, y1, x2, y2) in cropped car
    # }
```

**Usage:**
```python
plate_results = plate_detector.detect_plate(car_crop, offset=(car_x1, car_y1))
# plate_results[0]['bbox'] is now in original image coordinates!
```

### 2. `main.py` - Updated `process_image()`
**Now passes offset when detecting plates:**

```python
plate_detected = self.plate_detector.detect_plate(
    crop_car, 
    offset=(x1_car, y1_car)  # ← This converts coordinates!
)

for plate_info in plate_detected:
    bbox_original = plate_info['bbox']  # ✓ Ready to use in original image!
```

### 3. `src/detect_car.py` - Updated method signature
**Changed to accept both image paths and frames:**

```python
def detect_car(self, input_data) -> list:
    # Now works with:
    # - Image path: "path/to/image.jpg"
    # - Numpy array: cv2.imread(...) or cropped frame
```

## Real-World Example

```python
# Image is 1920x1080
image = cv2.imread("car.jpg")

# Detect car at position (100, 50) to (500, 400)
car_bbox = [
    (100, 50),      # top-left
    (500, 50),      # top-right
    (500, 400),     # bottom-right
    (100, 400)      # bottom-left
]

# Crop the car
crop_car = image[50:400, 100:500]  # shape is (350, 400)

# Detect plate in car crop - plate found at (50, 30, 150, 70)
plates = plate_detector.detect_plate(crop_car, offset=(100, 50))

# Result:
# plates[0]['bbox'] = (150, 80, 250, 120)  # In ORIGINAL 1920x1080 image!
# plates[0]['bbox_in_crop'] = (50, 30, 150, 70)  # In cropped 350x400 car

# Draw on original:
x1, y1, x2, y2 = plates[0]['bbox']
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# This works perfectly! The rectangle is at the right position
```

## Common Mistakes to Avoid

❌ **Don't** forget the offset:
```python
# Wrong - coordinates are relative to crop!
plates = plate_detector.detect_plate(crop_car)
bbox = plates[0]['bbox_in_crop']
cv2.rectangle(image, ...)  # Won't be in right position
```

❌ **Don't** mix coordinate spaces:
```python
# Wrong - mixing crop and original coordinates
x1_crop, y1_crop = 50, 30
x1_orig = x1_crop + 100  # ✓ Good
y1_orig = y1_crop + 30   # ✗ Bad! Should be + 50 (y-offset)
```

❌ **Don't** forget OpenCV indexing order:
```python
# OpenCV: image[y:y2, x:x2] (rows, then columns)
# Bboxes: (x1, y1, x2, y2) (x/column first, then y/row)

# Offset should be (x_offset, y_offset)
offset = (x1_crop_start, y1_crop_start)  # ✓ Correct

offset = (y1_crop_start, x1_crop_start)  # ✗ Wrong!
```

✅ **Do** keep three representations:
```python
# Original image coords - for drawing on full image
bbox_original = (150, 80, 250, 120)

# Crop coords - for drawing on cropped region
bbox_in_crop = (50, 30, 150, 70)

# Offset - for conversion between them
offset = (100, 50)

# Verify: bbox_in_crop + offset = bbox_original
assert (50 + 100, 30 + 50, 150 + 100, 70 + 50) == bbox_original
```

## Testing Your Implementation

```python
# Quick test
import cv2
from src.detect_car import CarDetector
from src.detect_plate import PlateDetector

car_det = CarDetector("yolo26n.pt")
plate_det = PlateDetector("yolo26_coco128_plate/best.pt")

image = cv2.imread("test.jpg")
cars = car_det.detect_car(image)

for car in cars:
    x1, y1, x2, y2 = car[0][0], car[0][1], car[2][0], car[2][1]
    crop = image[y1:y2, x1:x2]
    
    plates = plate_det.detect_plate(crop, offset=(x1, y1))
    
    for p in plates:
        bbox = p['bbox']  # Should be in original image space
        
        # Verify by drawing
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

cv2.imshow("Test", image)
cv2.waitKey(0)
```

---

**Files Updated:**
- ✅ `src/detect_plate.py` - Automatic offset-based conversion
- ✅ `main.py` - Passes offset parameter
- ✅ `src/detect_car.py` - Accepts image or frame input

**Example Files Created:**
- 📄 `BOUNDING_BOX_CONVERSION_GUIDE.md` - Detailed explanation with visuals
- 📄 `example_bbox_conversion.py` - Complete working example

**Summary:** Always add the offset to cropped coordinates to get original image coordinates!

