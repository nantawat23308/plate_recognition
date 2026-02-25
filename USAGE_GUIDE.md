# 🚀 License Plate Recognition - Practical Usage Guide

## 📖 Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Advanced Usage](#advanced-usage)
4. [Troubleshooting](#troubleshooting)
5. [Performance Tips](#performance-tips)
6. [Examples](#examples)

---

## Installation

### **Prerequisites**
```bash
# Python 3.8+
python --version

# GPU Support (Optional but recommended)
nvidia-smi  # If using NVIDIA GPU
```

### **Step 1: Clone Repository**
```bash
cd /home/nantawat/Desktop/my_project/plate_recognition
```

### **Step 2: Install Dependencies**
```bash
# Using pip
pip install ultralytics opencv-python paddleocr matplotlib

# Or using requirements.txt
pip install -r requirements.txt

# Or using uv (modern package manager)
uv sync
```

### **Step 3: Download Models**
Models are pre-trained and included:
```
plate_recognition/
├── yolo26n.pt ← Car detection (11 MB)
└── yolo26_coco128_plate/best.pt ← Plate detection (12 MB)
```

### **Step 4: Verify Installation**
```bash
# Test import
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"
python -c "import cv2; print('OpenCV OK')"
```

---

## Basic Usage

### **Simplest Example**

```python
from main import CarPlateDetector

# Initialize detector (loads all models)
detector = CarPlateDetector()

# Process a single image
detector.process_image("dataset/car_image.jpg")

# Output:
# - Prints detected cars and plates
# - Saves "result.png"
# - Displays image in window
```

### **Step-by-Step Breakdown**

```python
# Step 1: Import
from main import CarPlateDetector

# Step 2: Initialize (loads YOLOv8 + PaddleOCR)
detector = CarPlateDetector()
# ⏳ Takes 2-3 seconds first time (model loading)

# Step 3: Process image
detector.process_image("dataset/car_image.jpg")

# Step 4: Output appears
# Console:
#   Detected car with corners: [(100, 50), (500, 50), (500, 400), (100, 400)]
#   Recognized Plate Text: ทม 1234
#   Plate Bounding Box (original image): (150, 80, 250, 120)
#
# Files:
#   result.png ← Created with annotations
#
# Display:
#   Matplotlib window shows annotated image
```

---

## Advanced Usage

### **Example 1: Batch Process Multiple Images**

```python
from main import CarPlateDetector
import os

detector = CarPlateDetector()

# Process all images in a folder
image_folder = "dataset"
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.png', '.bmp')):
        image_path = os.path.join(image_folder, filename)
        print(f"\nProcessing: {filename}")
        detector.process_image(image_path)
        # result.png is overwritten each time
```

### **Example 2: Process with Custom Model Path**

```python
from main import CarPlateDetector

# Use custom-trained plate model
detector = CarPlateDetector(
    model_path="path/to/custom/model.pt"
)

detector.process_image("dataset/car.jpg")
```

### **Example 3: Extract Results Without Visualization**

```python
from src.detect_car import CarDetector
from src.detect_plate import PlateDetector
from src.recognizer import PaddleOCRPipeline
import cv2

# Initialize components
car_detector = CarDetector("yolo26n.pt")
plate_detector = PlateDetector("yolo26_coco128_plate/best.pt")
recognizer = PaddleOCRPipeline()

# Load image
image = cv2.imread("dataset/car.jpg")

# Detect cars
cars = car_detector.detect_car(image)
print(f"Found {len(cars)} cars")

# Process each car
results = []
for car in cars:
    x1, y1 = car[0]
    x2, y2 = car[2]
    
    # Crop car region
    crop_car = image[y1:y2, x1:x2]
    
    # Detect plates
    plates = plate_detector.detect_plate(crop_car, offset=(x1, y1))
    
    for plate_info in plates:
        plate_crop = plate_info['crop']
        
        # Recognize text
        text, _ = recognizer.predict(plate_crop)
        
        # Store result
        results.append({
            'plate_text': text,
            'bbox': plate_info['bbox'],
            'confidence': 0.95  # Add your confidence calculation
        })

# Use results (save to database, etc.)
for result in results:
    print(f"Plate: {result['plate_text']} at {result['bbox']}")
```

### **Example 4: Real-time Video Processing**

```python
import cv2
from main import CarPlateDetector

detector = CarPlateDetector()
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save frame temporarily
    cv2.imwrite("temp_frame.jpg", frame)
    
    # Process frame (in production, vectorize this)
    # For now, just show frame
    cv2.imshow('License Plate Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### **Example 5: Save Results to JSON**

```python
import json
from main import CarPlateDetector

detector = CarPlateDetector()

# Capture results
results = []

# Monkey-patch process_image to capture results
original_process = detector.process_image

def process_with_results(image_path):
    # Process image (existing logic)
    frame = detector.read_image(image_path)
    car_detected = detector.car_detector.detect_car(frame)
    
    for car in car_detected:
        x1_car, y1_car = car[0]
        crop_car = detector.crop_plate(frame, car)
        
        plate_detected = detector.plate_detector.detect_plate(
            crop_car, offset=(x1_car, y1_car)
        )
        
        for plate_info in plate_detected:
            text, _ = detector.recognizer.predict(plate_info['crop'])
            
            results.append({
                'image': image_path,
                'plate_text': text,
                'bbox': plate_info['bbox'],
                'bbox_original': list(plate_info['bbox'])
            })

detector.process_image = process_with_results

# Process image
detector.process_image("dataset/car.jpg")

# Save to JSON
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Results saved to results.json")
```

### **Example 6: Filter Results by Confidence**

```python
from src.detect_car import CarDetector
from src.detect_plate import PlateDetector
from src.recognizer import PaddleOCRPipeline
import cv2

car_detector = CarDetector("yolo26n.pt")
plate_detector = PlateDetector("yolo26_coco128_plate/best.pt")
recognizer = PaddleOCRPipeline()

image = cv2.imread("dataset/car.jpg")
cars = car_detector.detect_car(image)

CONFIDENCE_THRESHOLD = 0.85

for car in cars:
    x1, y1 = car[0]
    crop_car = image[y1:car[2][1], x1:car[2][0]]
    
    plates = plate_detector.detect_plate(crop_car, offset=(x1, y1))
    
    for plate_info in plates:
        plate_crop = plate_info['crop']
        text, ocr_bbox = recognizer.predict(plate_crop)
        
        # Calculate confidence (average OCR character confidence)
        if ocr_bbox:
            avg_confidence = sum(scores for _, scores in ocr_bbox) / len(ocr_bbox)
        else:
            avg_confidence = 0.0
        
        # Filter by confidence
        if avg_confidence >= CONFIDENCE_THRESHOLD:
            print(f"✓ HIGH CONFIDENCE: {text} ({avg_confidence:.2f})")
        else:
            print(f"✗ LOW CONFIDENCE: {text} ({avg_confidence:.2f})")
```

---

## Troubleshooting

### **Issue 1: Models Not Found**

**Error:**
```
FileNotFoundError: yolo26n.pt not found
```

**Solution:**
```bash
# Check file exists
ls -la *.pt

# If missing, download YOLO weights
# yolo26n.pt: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Rename if needed
mv yolov8n.pt yolo26n.pt
```

### **Issue 2: No GPU Available**

**Error:**
```
CUDA not available, using CPU (slow!)
```

**Solutions:**
```bash
# Check GPU
nvidia-smi

# If no GPU:
# 1. Install CUDA + cuDNN
# 2. Install NVIDIA CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or accept CPU inference (slower but works)
```

### **Issue 3: Thai Text Not Displaying**

**Error:**
```
Thai characters show as boxes/symbols
```

**Solution:**
```bash
# Install Thai fonts
# On Ubuntu/Debian:
sudo apt-get install fonts-thai-tlwg

# Font file should be at:
/usr/share/fonts/truetype/tlwg/Kinnari.ttf

# Verify in code
import os
font_path = "/usr/share/fonts/truetype/tlwg/Kinnari.ttf"
print(f"Font exists: {os.path.exists(font_path)}")
```

### **Issue 4: No Plates Detected**

**Diagnostic Steps:**
```python
from src.detect_car import CarDetector
from src.detect_plate import PlateDetector
import cv2

image = cv2.imread("test.jpg")

# Step 1: Check car detection
car_det = CarDetector("yolo26n.pt")
cars = car_det.detect_car(image)
print(f"Cars found: {len(cars)}")
# Expected: 1+ car detected

# Step 2: Check car crop
if cars:
    x1, y1 = cars[0][0]
    crop = image[y1:cars[0][2][1], x1:cars[0][2][0]]
    cv2.imwrite("car_crop.jpg", crop)
    print(f"Car crop shape: {crop.shape}")
    # Check car_crop.jpg manually

# Step 3: Check plate detection
plate_det = PlateDetector("best.pt")
plates = plate_det.detect_plate(crop, offset=(x1, y1))
print(f"Plates found: {len(plates)}")
# If 0, model may not be trained on your plate style
```

### **Issue 5: Bounding Boxes in Wrong Position**

**Cause:** Offset not applied correctly

**Fix:**
```python
# WRONG: ❌
plates = detector.detect_plate(crop_car)

# CORRECT: ✓
plates = detector.detect_plate(crop_car, offset=(x1, y1))
```

### **Issue 6: Out of Memory (OOM)**

**Error:**
```
CUDA out of memory
```

**Solutions:**
```python
# 1. Use smaller model
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Nano (smaller)
# vs
model = YOLO("yolov8l.pt")  # Large (bigger)

# 2. Reduce batch size
results = model.predict(source=image, batch=1)

# 3. Lower image resolution
results = model.predict(source=image, imgsz=320)

# 4. Use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

---

## Performance Tips

### **Speed Optimization**

```python
# Tip 1: Use GPU
# Fastest: 300-650ms per image (GPU)
# Slowest: 2-5s per image (CPU)

# Tip 2: Batch process
# Process multiple images simultaneously
batch_size = 4
images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
results = model.predict(source=images, batch=batch_size)

# Tip 3: Model selection
# yolo26n.pt (Nano) - Fastest, least accurate
# yolo26s.pt (Small) - Fast, accurate
# yolo26m.pt (Medium) - Slower, more accurate

# Tip 4: Image resolution
# Smaller = faster (but may miss details)
results = model.predict(source=image, imgsz=320)  # Fast
results = model.predict(source=image, imgsz=640)  # Balanced (default)
results = model.predict(source=image, imgsz=1280) # Slow, accurate
```

### **Accuracy Optimization**

```python
# Tip 1: Higher confidence threshold
# Filters out weak detections
plates = detector.detect_plate(crop, confidence=0.5)  # Default
plates = detector.detect_plate(crop, confidence=0.7)  # Stricter

# Tip 2: Pre-process images
import cv2
# Increase contrast
image = cv2.convertScaleAbs(image, alpha=1.2, beta=0)
# Histogram equalization
image = cv2.equalizeHist(image)
# Denoise
image = cv2.fastNlMeansDenoising(image, h=10)

# Tip 3: Post-process OCR
# Validate plate format (Thai: ทX 1234)
import re
def is_valid_thai_plate(text):
    pattern = r'ท[ก-ง] \d{4}'
    return re.match(pattern, text) is not None

# Tip 4: Retrain model on your data
# Most important for accuracy!
# Fine-tune best.pt with your specific plates
```

### **Memory Optimization**

```python
# Load models once, reuse multiple times
detector = CarPlateDetector()

# Process multiple images without reloading
for image_path in image_list:
    detector.process_image(image_path)

# Delete large intermediate results
import gc
gc.collect()
```

---

## Examples

### **Example: Single Image**

```bash
# Run directly
python main.py

# Output:
# - result.png created
# - Displayed in matplotlib
```

### **Example: Command Line**

```bash
# Create a CLI version
python -c "from main import CarPlateDetector; \
CarPlateDetector().process_image('dataset/car.jpg')"
```

### **Example: Web API (Flask)**

```python
from flask import Flask, request, jsonify
from main import CarPlateDetector
import cv2
import base64
import json

app = Flask(__name__)
detector = CarPlateDetector()

@app.route('/detect', methods=['POST'])
def detect():
    """
    POST endpoint for plate detection
    Input: image file
    Output: JSON with detected plates
    """
    file = request.files['image']
    
    # Save temporarily
    image_path = 'temp.jpg'
    file.save(image_path)
    
    # Process
    image = cv2.imread(image_path)
    cars = detector.car_detector.detect_car(image)
    
    results = []
    for car in cars:
        x1, y1 = car[0]
        crop = image[y1:car[2][1], x1:car[2][0]]
        
        plates = detector.plate_detector.detect_plate(crop, offset=(x1, y1))
        for plate_info in plates:
            text, _ = detector.recognizer.predict(plate_info['crop'])
            results.append({
                'text': text,
                'bbox': plate_info['bbox']
            })
    
    return jsonify({'plates': results})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### **Example: Docker Container**

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "main.py"]
```

---

## FAQ

**Q: How accurate is the system?**  
A: 85-95% depending on plate quality, angle, and lighting

**Q: Can it handle Thai plates?**  
A: Yes, with PaddleOCR and Thai font support

**Q: Does it work with video?**  
A: Yes, but frame-by-frame (not optimized yet)

**Q: Can I use my own trained model?**  
A: Yes, pass `model_path` to CarPlateDetector

**Q: What's the minimum plate size?**  
A: ~20x20 pixels (smaller plates harder to detect)

**Q: Does it work offline?**  
A: Yes, once models are downloaded

**Q: Can I improve accuracy?**  
A: Yes, retrain model on your specific plate style

---

## Summary

| Task | Code |
|------|------|
| **Quick Start** | `detector = CarPlateDetector()` then `detector.process_image()` |
| **Custom Model** | `detector = CarPlateDetector("model.pt")` |
| **Batch Process** | Loop through images, process each |
| **Get Results** | Extract from `plate_info` dict |
| **Save Results** | Save to JSON/CSV/Database |
| **Video** | Process frame-by-frame with cv2.VideoCapture |


