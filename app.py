from flask import Flask, request, jsonify
from main import CarPlateDetector
import cv2
app = Flask(__name__)
detector = CarPlateDetector()


@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    file.save('temp.jpg')

    # Process
    image = cv2.imread('temp.jpg')
    image = detector.process_image(image)
    # Convert processed image to bytes for response
    unicode64 = cv2.imencode('.jpg', image)[1].tobytes()

    return jsonify({'unicode64': unicode64})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)