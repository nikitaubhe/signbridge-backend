from flask import Flask, request, jsonify
from flask_cors import CORS
from function import * 
from keras.models import model_from_json
import cv2
import numpy as np
import mediapipe as mp
import base64
from io import BytesIO
from PIL import Image

# Define actions (labels) matching the model output
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F'])

app = Flask(__name__)
CORS(app)

# Load trained model
print("Loading model...")
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")
print("Model loaded successfully!")

# Word mapping
word_map = {
    'A': 'Hello',
    'B': 'Yes',
    'C': 'No',
    'D': 'Thank You',
    'E': 'I Love You',
    'F': 'See You Again'
}

# Global variables for sequence tracking
sequence = []
predictions = []
threshold = 0.8

# MediaPipe is initialized in function.py


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'message': 'Python ML service is running'
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    global sequence, predictions

    try:
        data = request.json
        frame_data = data.get('frame')

        if not frame_data:
            return jsonify({
                'success': False,
                'message': 'No frame data provided'
            }), 400

        # Decode base64 image
        if 'base64,' in frame_data:
            frame_data = frame_data.split('base64,')[1]

        image_bytes = base64.b64decode(frame_data)
        image = Image.open(BytesIO(image_bytes))
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Process frame with MediaPipe (detector is global)
        image, results = mediapipe_detection(frame, detector)
        keypoints = extract_keypoints(results)

        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep last 30 frames

        if len(sequence) == 30:
            # Make prediction
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_idx = np.argmax(res)
            predicted_action = actions[predicted_idx]
            confidence = float(res[predicted_idx])

            predictions.append(predicted_idx)

            # Stabilize prediction
            if len(predictions) >= 10:
                if np.unique(predictions[-10:])[0] == predicted_idx and confidence > threshold:
                    display_word = word_map.get(predicted_action, predicted_action)

                    return jsonify({
                        'success': True,
                        'predictedSign': predicted_action,
                        'mappedWord': display_word,
                        'confidence': confidence,
                        'message': 'Prediction successful'
                    }), 200

            return jsonify({
                'success': True,
                'predictedSign': predicted_action,
                'mappedWord': word_map.get(predicted_action, predicted_action),
                'confidence': confidence,
                'message': 'Prediction in progress (stabilizing...)',
                'requiresMoreFrames': True
            }), 200
        else:
            return jsonify({
                'success': True,
                'message': f'Building sequence... {len(sequence)}/30 frames',
                'requiresMoreFrames': True
            }), 200

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500


@app.route('/reset', methods=['POST'])
def reset_sequence():
    global sequence, predictions
    sequence = []
    predictions = []
    return jsonify({
        'success': True,
        'message': 'Sequence reset successfully'
    }), 200


@app.route('/actions', methods=['GET'])
def get_actions():
    return jsonify({
        'actions': actions.tolist(),
        'wordMap': word_map
    }), 200


if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
