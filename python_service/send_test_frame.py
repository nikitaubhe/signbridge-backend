import requests
import json
import base64
import sys

# Create a small valid 1x1 image in base64
b64_image = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="

frame_data = "data:image/jpeg;base64," + b64_image

print("Sending POST request to http://localhost:5000/predict ...")
try:
    response = requests.post("http://localhost:5000/predict", json={"frame": frame_data}, timeout=10)
    print("Status Code:", response.status_code)
    try:
        print("JSON Response:", json.dumps(response.json(), indent=2))
    except:
        print("Raw Response:", response.text)
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
    sys.exit(1)
