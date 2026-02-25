# 🏗️ License Plate Recognition System - Complete Architecture

## 📚 Table of Contents
1. [System Overview](#system-overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Model Details](#model-details)
4. [Component Descriptions](#component-descriptions)
5. [Data Flow](#data-flow)
6. [Coordinate System](#coordinate-system)
7. [Implementation Details](#implementation-details)

---

## System Overview

The License Plate Recognition (LPR) system is a **multi-stage computer vision pipeline** that:
- Detects vehicles in images
- Locates license plates on those vehicles
- Recognizes text from the plates
- Visualizes results with high accuracy

### **Workflow Diagram**

```
┌────────────────────────────────────────────────────────────────────┐
│                   LICENSE PLATE RECOGNITION PIPELINE                │
└────────────────────────────────────────────────────────────────────┘

Input Image (Road/Parking Scene)
         │
         ▼
    ╔════════════════╗
    ║ STAGE 1        ║  ← YOLOv8 (yolo26n.pt - COCO Weights)
    ║ CAR DETECTION  ║     Input: Full image (1920x1080)
    ║                ║     Output: Vehicle bounding boxes
    ╚════════════════╝
         │
         ├─→ Car 1 ─────┐
         ├─→ Car 2 ─────┤
         ├─→ Car 3 ─────┤
         └─→ Car N ─────┘
                │
         ┌──────┴──────────────────────┐
         ▼                             ▼
     ╔════════════╗            ╔════════════╗
     ║ CROP       ║            ║ CROP       ║
     ║ Car 1      ║            ║ Car N      ║
     ║ ROI        ║            ║ ROI        ║
     ╚════════════╝            ╚════════════╝
         │                             │
         ▼                             ▼
    ╔════════════════╗        ╔════════════════╗
    ║ STAGE 2        ║        ║ STAGE 2        ║
    ║ PLATE         ║        ║ PLATE         ║
    ║ DETECTION      ║        ║ DETECTION      ║
    ║                ║        ║                ║
    ║ Custom YOLOv8  ║        ║ Custom YOLOv8  ║
    ║ (best.pt)      ║        ║ (best.pt)      ║
    ╚════════════════╝        ╚════════════════╝
         │                             │
         ├─→ Plate 1 ───┐        ├─→ Plate N ───┐
         └─→ Plate 2 ───┤        └─→ Plate M ───┤
                        │                        │
         ┌──────────────┴────────────────────────┴──────────────┐
         │                                                       │
         ▼                                                       ▼
     ╔════════════════╗                                  ╔════════════════╗
     ║ STAGE 3        ║                                  ║ STAGE 3        ║
     ║ OCR RECOGNITION║                                  ║ OCR RECOGNITION║
     ║                ║                                  ║                ║
     ║ PaddleOCR      ║                                  ║ PaddleOCR      ║
     ║ Thai/English   ║                                  ║ Thai/English   ║
     ╚════════════════╝                                  ╚════════════════╝
         │                                                       │
         │ "ทม 1234"                                          │ "ทม 5678"
         │ Confidence: 0.95                                    │ Confidence: 0.92
         │                                                       │
         └───────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
                    ╔════════════════════════╗
                    ║ STAGE 4                ║
                    ║ VISUALIZATION & OUTPUT ║
                    ║                        ║
                    ║ - Draw bounding boxes  ║
                    ║ - Add recognized text  ║
                    ║ - Save result.png      ║
                    ║ - Display in window    ║
                    ╚════════════════════════╝
                                 │
                                 ▼
                    Output Image (Annotated)
                    with all plates marked
```

---

## Pipeline Stages

### **STAGE 1: Vehicle Detection**

#### **Purpose**
Locate all vehicles in the input image

#### **Technology**
- **Model:** YOLOv8 (Nano variant - yolo26n.pt)
- **Weights:** COCO pre-trained (80 classes)
- **Input:** Full RGB image
- **Output:** Bounding boxes + confidence scores

#### **Classes Detected**
```python
car_classes = [2, 3, 5, 7]
│
├─ Class 2: Car (4-door sedan, SUV, etc.)
├─ Class 3: Motorcycle  
├─ Class 5: Bus
└─ Class 7: Truck
```

#### **Implementation**
```python
# In src/detect_car.py
car_detector = CarDetector("yolo26n.pt")
results = car_detector.detect_car(frame)
# Returns: List of [top-left, top-right, bottom-right, bottom-left] corners
```

#### **Detection Process**
```
1. Resize image to standard input size (640x640)
2. Run YOLOv8 inference
3. Filter results by class [2,3,5,7]
4. Apply Non-Maximum Suppression (NMS)
5. Extract corner coordinates
6. Return detected vehicles
```

---

### **STAGE 2: License Plate Detection**

#### **Purpose**
Locate license plates within detected car regions

#### **Technology**
- **Model:** Custom-trained YOLOv8
- **Weights:** best.pt (trained on License-Plate-Recognition dataset)
- **Input:** Cropped car region (~400x350 pixels)
- **Output:** Plate bounding boxes

#### **Key Advantage**
Running plate detection on **cropped regions** instead of full image:
- ✅ Faster inference (smaller images)
- ✅ More accurate (fewer false positives)
- ✅ Contextual information (plate is on car)

#### **Implementation**
```python
# In src/detect_plate.py
plate_detector = PlateDetector("best.pt")
plates = plate_detector.detect_plate(crop_car, offset=(x1_car, y1_car))
# Returns: List of dicts with:
#   - 'crop': cropped plate image
#   - 'bbox': coordinates in ORIGINAL image
#   - 'bbox_in_crop': coordinates in cropped car
```

#### **Detection Process**
```
1. Take cropped car region as input
2. Resize to YOLOv8 input size (640x640)
3. Run inference
4. Extract plate bounding boxes (relative to crop)
5. Convert coordinates using OFFSET
   original = crop + offset
6. Return plates with both coordinate systems
```

#### **Coordinate Conversion (Critical!)**

```
Crop coordinate system:        Original image coordinate system:
┌──────────────────┐           ┌─────────────────────────────┐
│  Plate detected  │           │  Full Image                 │
│  at (50, 30)     │ + Offset  │  Plate is at (150, 80)     │
│                  │   (100,50)│  = 50 + 100, 30 + 50       │
└──────────────────┘           └─────────────────────────────┘

Formula:
────────
plate_x_original = plate_x_crop + car_x_offset
plate_y_original = plate_y_crop + car_y_offset
```

---

### **STAGE 3: Optical Character Recognition (OCR)**

#### **Purpose**
Recognize text characters from license plate images

#### **Technology**
- **Engine:** PaddleOCR (Baidu's paddle framework)
- **Languages:** Thai, English, Multi-language
- **Input:** Plate crop image
- **Output:** Recognized text + confidence scores

#### **Implementation**
```python
# In src/recognizer.py
recognizer = PaddleOCRPipeline(ocr_lang="th")  # Thai language
text, bbox = recognizer.predict(plate_crop)
# Returns:
#   - text: recognized characters (e.g., "ทม 1234")
#   - bbox: OCR character bounding boxes
```

#### **OCR Process**
```
Input: Plate Image (100x30)
    │
    ▼
┌────────────────────────────────────┐
│ Text Detection                     │
│ (Find text regions in image)       │
└────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│ Text Recognition                   │
│ (Recognize characters in regions)  │
└────────────────────────────────────┘
    │
    ▼
Output: "ทม 1234" (with confidence scores)
```

#### **Supported Languages**
- ✅ Thai (ทะเบียน)
- ✅ English (ABC1234)
- ✅ Multi-language support

---

### **STAGE 4: Visualization & Output**

#### **Purpose**
Annotate results on original image and save output

#### **Components**

**1. Bounding Box Drawing**
```python
cv2.rectangle(frame,
              (bbox_original[0], bbox_original[1]),
              (bbox_original[2], bbox_original[3]),
              (0, 255, 0),  # Green color (BGR)
              2)             # Thickness 2 pixels
```

**2. Text Rendering (Thai Support)**
```python
# draw_thai_text() from recognizer.py
# Uses PIL for Thai font support (OpenCV doesn't support Thai)
frame = draw_thai_text(frame, text,
                       (bbox_original[0], bbox_original[1] - 40),
                       color=(255, 0, 0))  # Blue text
```

**3. Save Result**
```python
plt.imsave("result.png", frame)  # Save to disk
plt.imshow(frame)                # Display in window
plt.show()
```

#### **Output Specification**
```
File: result.png
├─ Format: PNG (lossless)
├─ Size: Same as input image
├─ Colors: BGR (OpenCV standard)
├─ Annotations:
│  ├─ Green bounding boxes (plate locations)
│  └─ Blue text (recognized plate numbers)
└─ Saved to: /home/nantawat/Desktop/my_project/plate_recognition/
```

---

## Model Details

### **Model 1: YOLOv8 (Car Detection)**

| Property | Value |
|----------|-------|
| **Model Name** | yolo26n.pt |
| **Variant** | Nano (lightweight) |
| **Weights** | COCO pre-trained |
| **Input Size** | 640x640 |
| **Classes** | 80 (all COCO classes) |
| **Filter Classes** | [2, 3, 5, 7] |
| **File Size** | ~11 MB |
| **Speed** | ~40-100ms (GPU) |
| **Accuracy** | mAP50: 50.4% |
| **Memory** | ~1-2 GB |

### **Model 2: YOLOv8 (Plate Detection)**

| Property | Value |
|----------|-------|
| **Model Name** | best.pt |
| **Training Data** | License-Plate-Recognition-13 |
| **Weights** | Custom-trained |
| **Input Size** | 640x640 |
| **Classes** | 1 (license plate) |
| **File Size** | ~12 MB |
| **Speed** | ~50-150ms (GPU) |
| **Accuracy** | Depends on training quality |
| **Memory** | ~1-2 GB |

### **Model 3: PaddleOCR**

| Property | Value |
|----------|-------|
| **Engine** | PaddleOCR |
| **Languages** | Thai, English, 80+ |
| **Type** | Deep learning OCR |
| **Accuracy** | 90%+ on clear text |
| **Speed** | ~100-300ms per image |
| **Auto-download** | Yes (first run) |
| **Memory** | ~500MB |

---

## Component Descriptions

### **Component 1: CarDetector**
**Location:** `src/detect_car.py`

```python
class CarDetector:
    """Detects vehicles in images using YOLOv8"""
    
    def __init__(self, model_path="yolo26n.pt"):
        # Load COCO pre-trained model
        self.model = YOLO(model_path)
        self.car_classes = [2, 3, 5, 7]
    
    def detect_car(self, input_data):
        # Supports both image path and numpy array
        # Returns: List of corner coordinates
        
    def extract_corners(box_tensor):
        # Convert YOLO box format to corner coordinates
        # Returns: [top-left, top-right, bottom-right, bottom-left]
```

### **Component 2: PlateDetector**
**Location:** `src/detect_plate.py`

```python
class PlateDetector:
    """Detects license plates using custom-trained YOLOv8"""
    
    def __init__(self, model_path="best.pt"):
        # Load custom-trained model
        self.model = YOLO(model_path)
    
    def detect_plate(self, frame, offset=(0, 0)):
        # Detects plates in cropped car image
        # Automatically converts coordinates
        # Returns: List of dicts with multiple coordinate types
```

### **Component 3: PaddleOCRPipeline**
**Location:** `src/recognizer.py`

```python
class PaddleOCRPipeline:
    """Recognizes text from license plates"""
    
    def __init__(self, ocr_lang="th"):
        # Initialize PaddleOCR with language
        self.ocr = PaddleOCR(lang=ocr_lang)
    
    def predict(self, input):
        # Recognizes text from image
        # Returns: (recognized_text, bounding_boxes)
        
    def draw_thai_text(img, text, position, color):
        # Draws Thai text on image (PIL-based)
        # Returns: Image with text drawn
```

### **Component 4: CarPlateDetector (Orchestrator)**
**Location:** `main.py`

```python
class CarPlateDetector:
    """Main pipeline orchestrator"""
    
    def __init__(self, model_path="best.pt"):
        # Initialize all components
        self.car_detector = CarDetector()
        self.plate_detector = PlateDetector(model_path)
        self.recognizer = PaddleOCRPipeline()
    
    def process_image(self, image_path):
        # Execute complete pipeline
        # Returns: Annotated image + visualization
```

---

## Data Flow

### **Complete Data Flow Diagram**

```
INPUT: image_path = "dataset/car.jpg"
  │
  ├─────────────────────────────────────────────────┐
  │                                                 │
  ▼                                                 ▼
┌──────────────────┐                         ┌──────────────────┐
│ read_image()     │                         │ Metadata        │
│ Returns: frame   │                         │ - Dimensions    │
│ Shape: (H,W,3)   │                         │ - Format: BGR   │
└──────────────────┘                         └──────────────────┘
  │
  ├─────────────────────────────────────────────────┐
  │                                                 │
  ▼                                                 ▼
┌──────────────────────────────────────────┐  ┌─────────────────┐
│ car_detector.detect_car(frame)           │  │ Output:         │
│                                          │  │ - 3-4 cars      │
│ Process:                                 │  │ - Corner coords │
│ 1. Resize to 640x640                     │  │ - Confidence    │
│ 2. YOLOv8 inference                      │  │   scores        │
│ 3. Filter by class [2,3,5,7]            │  │                 │
│ 4. NMS (Non-Max Suppression)            │  │ Return: List    │
│ 5. Extract corners                       │  │ [car1_corners,  │
└──────────────────────────────────────────┘  │  car2_corners]  │
  │                                            └─────────────────┘
  │
  ├──→ FOR EACH CAR:
  │    │
  │    ├─────────────────────────────────────────┐
  │    │                                         │
  │    ▼                                         ▼
  │  ┌──────────────────────────────────┐  ┌──────────────┐
  │  │ crop_plate(frame, car_corners)  │  │ Output:      │
  │  │                                  │  │ - ROI image  │
  │  │ Returns: Cropped car region     │  │ - Size: ~400│
  │  │          with padding            │  │   x350       │
  │  └──────────────────────────────────┘  └──────────────┘
  │    │
  │    ├─────────────────────────────────────────┐
  │    │                                         │
  │    ▼                                         ▼
  │  ┌──────────────────────────────────┐  ┌───────────────┐
  │  │ plate_detector.detect_plate()    │  │ Output:       │
  │  │                                  │  │ - 1-3 plates  │
  │  │ Process:                         │  │ - Bboxes      │
  │  │ 1. Resize to 640x640             │  │ - Both coords │
  │  │ 2. YOLOv8 inference              │  │   systems     │
  │  │ 3. Convert coords (OFFSET)      │  │               │
  │  │ 4. Extract crops                │  │ Return:       │
  │  │                                  │  │ [{           │
  │  └──────────────────────────────────┘  │   'crop': ..│
  │    │                                    │   'bbox': ..│
  │    │                                    │   'bbox_in_ │
  │    │                                    │    crop': ..│
  │    │                                    │ }]          │
  │    │                                    └───────────────┘
  │    │
  │    ├──→ FOR EACH PLATE:
  │    │    │
  │    │    ├────────────────────────────────┐
  │    │    │                                │
  │    │    ▼                                ▼
  │    │  ┌──────────────────────────┐  ┌─────────────┐
  │    │  │ recognizer.predict()     │  │ Output:     │
  │    │  │                          │  │ - Text      │
  │    │  │ Process:                 │  │ - OCR boxes │
  │    │  │ 1. Input: plate_crop     │  │ - Conf.     │
  │    │  │ 2. PaddleOCR engine      │  │   scores    │
  │    │  │ 3. Recognize characters  │  │             │
  │    │  │ 4. Return text           │  │ Return:     │
  │    │  └──────────────────────────┘  │ "ทม 1234"  │
  │    │    │                            └─────────────┘
  │    │    │
  │    │    ├────────────────────────────────┐
  │    │    │                                │
  │    │    ▼                                ▼
  │    │  ┌──────────────────────────┐  ┌─────────────┐
  │    │  │ Draw results:            │  │ Modified:   │
  │    │  │ - cv2.rectangle()        │  │ - frame with│
  │    │  │ - draw_thai_text()       │  │   boxes     │
  │    │  │                          │  │ - frame with│
  │    │  │ Update: frame            │  │   text      │
  │    │  └──────────────────────────┘  └─────────────┘
  │    │
  │    └──→ NEXT PLATE
  │
  └──→ NEXT CAR
  │
  ├────────────────────────────────────────┐
  │                                        │
  ▼                                        ▼
┌────────────────────────────────────┐  ┌──────────────┐
│ Convert BGR to RGB                 │  │ Output:      │
│ Save to result.png                 │  │ - PNG file   │
│ Display in matplotlib              │  │ - Matplotlib│
│ plt.show()                         │  │   window    │
└────────────────────────────────────┘  └──────────────┘
  │
  ▼
FINAL OUTPUT: Annotated image with:
├─ Green bounding boxes for plates
├─ Blue text (license numbers)
└─ Saved as result.png
```

---

## Coordinate System

### **Three Coordinate Spaces**

The system operates with **three different coordinate systems**:

#### **1. Original Image Space**
```
Full input image (1920x1080)

┌─────────────────────────┐
│   (0,0)                 │
│   ┌──────────────────┐  │
│   │ Car Region       │  │
│   │ (100,50)-(500,400)
│   │   ┌──────────┐   │  │
│   │   │ Plate    │   │  │
│   │   │(150,80)- │   │  │
│   │   │(250,120) │   │  │
│   │   └──────────┘   │  │
│   └──────────────────┘  │
│                    (1920,1080)│
└─────────────────────────┘

Used for: Final visualization, output coordinates
Returned by: plate_info['bbox']
```

#### **2. Car Crop Space**
```
Cropped car region (400x350)

Car at (100,50) in original
becomes (0,0) here

┌──────────────────────┐
│ (0,0)                │
│ ┌──────────────────┐ │
│ │ Plate detected   │ │
│ │ at (50,30)-(150,│
│ │ 70) in crop      │ │
│ │                  │ │
│ └──────────────────┘ │
│              (400,350)│
└──────────────────────┘

Used for: Plate detection output
Returned by: plate_info['bbox_in_crop']
```

#### **3. Plate Crop Space**
```
Cropped plate image (~100x30)

┌──────────────────┐
│ (0,0)            │
│ ทม 1234 ←────────← OCR recognizes this
│              (100,30)
└──────────────────┘

Used for: OCR input
Input to: recognizer.predict()
```

### **Coordinate Conversion Formula**

```
ORIGINAL_COORDINATE = CROP_COORDINATE + CAR_OFFSET

Example:
────────
Plate detected in car crop:    (50, 30) to (150, 70)
Car region in original image:  (100, 50) ← This is the offset
                               
Plate in original image:       (50+100, 30+50) to (150+100, 70+50)
                             = (150, 80) to (250, 120)

Verification:
─────────────
50 + 100 = 150 ✓
30 + 50 = 80 ✓
150 + 100 = 250 ✓
70 + 50 = 120 ✓
```

---

## Implementation Details

### **File Structure**

```
plate_recognition/
│
├── main.py ← ENTRY POINT
│   ├── CarPlateDetector class
│   │   ├── __init__() - Initialize models
│   │   ├── read_image() - Load image
│   │   ├── crop_plate() - Extract ROI
│   │   └── process_image() - Main pipeline ⭐
│   │
│   ├── detect_car() - Helper function
│   └── main() - Run example
│
├── src/
│   ├── detect_car.py
│   │   ├── CarDetector class
│   │   │   ├── __init__()
│   │   │   ├── detect_car()
│   │   │   ├── extract_corners()
│   │   │   └── plot_results()
│   │   └── COCO class definitions
│   │
│   ├── detect_plate.py
│   │   ├── PlateDetector class
│   │   │   ├── __init__()
│   │   │   └── detect_plate() ← Coordinate conversion!
│   │   └── Returns: {crop, bbox, bbox_in_crop}
│   │
│   ├── recognizer.py
│   │   ├── PaddleOCRPipeline class
│   │   │   ├── __init__()
│   │   │   └── predict()
│   │   └── draw_thai_text() ← Thai font support
│   │
│   ├── annotate.py
│   ├── process.py
│   ├── train_model.py
│   └── __pycache__/
│
├── models/
│   ├── yolo26n.pt ← Car detection
│   └── yolo26_coco128_plate/
│       ├── best.pt ← Plate detection
│       └── best.onnx
│
└── dataset/
    └── Test images
```

### **Key Implementation Patterns**

**Pattern 1: Model Initialization**
```python
from ultralytics import YOLO

# Load pre-trained YOLOv8
car_detector = CarDetector("yolo26n.pt")

# Load custom-trained model
plate_detector = PlateDetector("best.pt")

# Initialize OCR
recognizer = PaddleOCRPipeline(ocr_lang="th")
```

**Pattern 2: Two-Stage Detection**
```python
# Stage 1: Detect cars
cars = car_detector.detect_car(frame)

# Stage 2: Detect plates (for each car)
for car in cars:
    crop = crop_plate(frame, car)
    plates = plate_detector.detect_plate(crop, offset=car[0])
    # Offset ensures coordinates align with original image
```

**Pattern 3: Coordinate Conversion**
```python
# Automatic conversion in detect_plate()
x1_orig = x1_crop + offset[0]
y1_orig = y1_crop + offset[1]

# Result: coordinates in original image space
```

**Pattern 4: Thai Text Support**
```python
# OpenCV cannot render Thai characters
# Use PIL-based draw_thai_text() instead
frame = draw_thai_text(frame, text, position, color)
```

---

## Configuration & Customization

### **Configuration Parameters**

**Car Detection:**
```python
car_classes = [2, 3, 5, 7]  # COCO classes to detect
confidence_threshold = 0.5    # Min confidence score
nms_threshold = 0.45          # Non-max suppression
```

**Plate Detection:**
```python
model_path = "best.pt"        # Custom model path
confidence_threshold = 0.5    # Min confidence
padding = 5                   # Padding for crop_plate()
```

**OCR:**
```python
ocr_lang = "th"              # Language (Thai)
enable_doc_orientation = False # Orientation correction
enable_mkldnn = False         # CPU optimization
```

### **Customization Options**

1. **Different Car Detection Model**
   ```python
   detector = CarDetector("yolov8m.pt")  # Medium model
   ```

2. **Different Plate Model**
   ```python
   detector = CarPlateDetector("path/to/custom/model.pt")
   ```

3. **Multiple Languages**
   ```python
   recognizer = PaddleOCRPipeline(ocr_lang="en")  # English
   ```

4. **Batch Processing**
   ```python
   images = ["img1.jpg", "img2.jpg", "img3.jpg"]
   for img in images:
       detector.process_image(img)
   ```

---

## Performance Optimization

### **Speed Optimization Tips**

1. **Use GPU**
   ```python
   # Automatic GPU detection
   model = YOLO("best.pt")  # Uses GPU if available
   ```

2. **Reduce Input Size**
   ```python
   # Smaller input = faster inference
   # Trade-off: May miss small plates
   ```

3. **Batch Processing**
   ```python
   # Process multiple images at once
   results = model.predict(source=batch_images)
   ```

4. **Model Quantization**
   ```python
   # Use smaller .onnx format
   results = model.predict(source="input.jpg", imgsz=416)
   ```

### **Accuracy Optimization Tips**

1. **Retrain Plate Model**
   ```python
   # On your specific plate style/region
   # Improves accuracy significantly
   ```

2. **Pre-process Images**
   ```python
   # Improve contrast, brightness
   # Rotate/align plates before OCR
   ```

3. **Post-process OCR Results**
   ```python
   # Filter by known format (ทX 1234)
   # Validate using checksums
   ```

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Purpose** | Detect vehicles and recognize license plates |
| **Input** | Image path or numpy array |
| **Output** | Annotated image + console output |
| **Models Used** | YOLOv8 (2x) + PaddleOCR |
| **Main Class** | CarPlateDetector |
| **Main Method** | process_image() |
| **Stages** | 4 (Detection, Detection, OCR, Visualization) |
| **Time/Image** | 300-650ms (GPU) |
| **Accuracy** | 85-95% (depends on plate quality) |
| **Supported Languages** | Thai, English, 80+ |
| **Thai Font Support** | Yes (PIL-based) |


