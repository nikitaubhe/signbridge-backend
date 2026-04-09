import requests
import cv2
import base64
import numpy as np

try:
    # Create a dummy image a black 640x480 RGB image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    frame_data = "data:image/jpeg;base64," + jpg_as_text

    print("Sending POST request to http://localhost:5000/predict...")
    response = requests.post("http://localhost:5000/predict", json={"frameData": frame_data}, timeout=30)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print("Error connecting to server:", e)
