from flask import Flask, request, jsonify
from flask_cors import CORS
from function import get_detector, extract_keypoints, mediapipe_detection
from detector_utils import detect_sign_heuristic
from keras.models import Sequential
from keras.layers import LSTM, Dense
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import logging
import os
 
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
 
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
 
# ── Actions (must match training order exactly) ──────────────────────────────
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
 
# Word mapping (Heuristic Keys)
word_map = {
    'HELLO':      'Hello 👋',
    'YES':        'Yes 👍',
    'NO':         'No 👎',
    'THANK_YOU':  'Thank You ❤️',
    'LOVE':       'I Love You 🤟',
    'OKAY':       'Okay 👌',
    'PEACE':      'Peace/Victory ✌️',
    'STOP':       'Stop/Wait ✋'
}
 
# ── Load model ───────────────────────────────────────────────────────────────
logger.info("Loading model...")
 
def build_keras_model():
    model_seq = Sequential()
    model_seq.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
    model_seq.add(LSTM(128, return_sequences=True, activation='relu'))
    model_seq.add(LSTM(64, return_sequences=False, activation='relu'))
    model_seq.add(Dense(64, activation='relu'))
    model_seq.add(Dense(32, activation='relu'))
    model_seq.add(Dense(6, activation='softmax'))
    return model_seq
 
try:
    model = build_keras_model()
    model.load_weights("model.h5")
    logger.info("Model loaded OK (Native Keras)")
except Exception as e:
    logger.error(f"Model load FAILED: {e}")
    model = None
 
# ── Parameters ───────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 30
THRESHOLD       = 0.6
 
# State (per-server; single user)
sequence    = []
predictions = []
 
 
@app.route("/")
def home():
    return "Backend is running 🚀"
 
 
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Python ML service is running'
    }), 200
 
 
@app.route('/predict', methods=['POST'])
def predict():
    global sequence, predictions
 
    if model is None:
        return jsonify({'success': False, 'message': 'Model not loaded'}), 500
 
    try:
        data = request.json
        frame_data = data.get('frame') or data.get('frameData')
 
        if not frame_data:
            return jsonify({'success': False, 'message': 'No frame data'}), 400
 
        # Strip base64 header
        if 'base64,' in frame_data:
            frame_data = frame_data.split('base64,')[1]
 
        # Decode image
        image_bytes = base64.b64decode(frame_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
 
        # MediaPipe detection
        _, results = mediapipe_detection(frame, get_detector())
        keypoints = extract_keypoints(results)
 
        hand_detected = bool(np.any(keypoints != 0))
 
        # ── 1. Heuristic Detection (Fast Overrides) ──
        h_sign, h_conf = detect_sign_heuristic(keypoints)
 
        if h_sign and h_conf > 0.8:
            display_word = word_map.get(h_sign, h_sign)
            return jsonify({
                'success':            True,
                'requiresMoreFrames': False,
                'predictedSign':      h_sign,
                'mappedWord':         display_word,
                'confidence':         h_conf,
                'handDetected':       hand_detected,
                'message':            'OK (Heuristic)'
            }), 200
 
        # ── 2. Sequential ML Detection (A-F Alphabet) ──
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]
 
        if len(sequence) < SEQUENCE_LENGTH:
            return jsonify({
                'success':            True,
                'requiresMoreFrames': True,
                'progress':           len(sequence),
                'total':              SEQUENCE_LENGTH,
                'handDetected':       hand_detected,
                'message':            f'Collecting {len(sequence)}/{SEQUENCE_LENGTH}'
            }), 200
 
        input_data = np.expand_dims(sequence, axis=0)
        res = model(input_data, training=False)[0]
        max_idx = int(np.argmax(res))
        conf = float(res[max_idx])
 
        predictions.append(max_idx)
        predictions = predictions[-10:]
 
        if conf > THRESHOLD:
            if predictions.count(max_idx) >= 4:
                predicted_action = str(actions[max_idx])
                return jsonify({
                    'success':            True,
                    'requiresMoreFrames': False,
                    'predictedSign':      predicted_action,
                    'mappedWord':         predicted_action,
                    'confidence':         conf,
                    'handDetected':       hand_detected,
                    'message':            'OK (ML Model)'
                }), 200
 
        return jsonify({
            'success':            True,
            'requiresMoreFrames': False,
            'predictedSign':      None,
            'confidence':         0.0,
            'handDetected':       hand_detected,
            'message':            'Detecting...'
        }), 200
 
    except Exception as e:
        logger.error(f"Predict error: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500
 
 
@app.route('/reset', methods=['POST'])
def reset_sequence():
    global sequence, predictions
    sequence    = []
    predictions = []
    return jsonify({'success': True, 'message': 'Reset OK'}), 200
 
 
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Flask starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=False)


