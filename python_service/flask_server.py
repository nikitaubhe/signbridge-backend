import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

import cv2
import numpy as np
import base64
import logging
from io import BytesIO
from collections import deque

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from keras.models import Sequential
from keras.layers import LSTM, Dense

from function import mediapipe_detection, extract_keypoints, detector
from detector_utils import detect_sign_heuristic

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── App & CORS ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ── Constants ─────────────────────────────────────────────────────────────────
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F'])

word_map = {
    'HELLO':     'Hello 👋',
    'YES':       'Yes 👍',
    'NO':        'No 👎',
    'THANK_YOU': 'Thank You ❤️',
    'LOVE':      'I Love You 🤟',
    'OKAY':      'Okay 👌',
    'PEACE':     'Peace/Victory ✌️',
    'STOP':      'Stop/Wait ✋',
}

SEQUENCE_LENGTH = 30
THRESHOLD       = 0.6

# ── Build & Load Keras Model (once at startup) ────────────────────────────────
def build_model():
    m = Sequential()
    m.add(LSTM(64,  return_sequences=True,  activation='relu', input_shape=(30, 63)))
    m.add(LSTM(128, return_sequences=True,  activation='relu'))
    m.add(LSTM(64,  return_sequences=False, activation='relu'))
    m.add(Dense(64, activation='relu'))
    m.add(Dense(32, activation='relu'))
    m.add(Dense(6,  activation='softmax'))
    return m

logger.info("Loading model...")
try:
    model = build_model()
    model.load_weights("model.h5")
    # Warm up the model so first request isn't slow
    dummy = np.zeros((1, 30, 63), dtype=np.float32)
    _ = model(dummy, training=False)
    logger.info("Model loaded OK ✅")
except Exception as e:
    logger.error(f"Model load FAILED: {e}")
    model = None

# ── Per-request state (single-user server) ────────────────────────────────────
sequence    = deque(maxlen=SEQUENCE_LENGTH)
predictions = []
h_history   = deque(maxlen=5)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({'status': 'running', 'model_loaded': model is not None}), 200


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status':       'healthy',
        'model_loaded': model is not None,
        'message':      'Python ML service is running'
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    global sequence, predictions, h_history

    if model is None:
        return jsonify({'success': False, 'message': 'Model not loaded'}), 500

    try:
        data = request.json
        frame_data = data.get('frame') or data.get('frameData')

        if not frame_data:
            return jsonify({'success': False, 'message': 'No frame data'}), 400

        # Strip base64 header if present
        if 'base64,' in frame_data:
            frame_data = frame_data.split('base64,')[1]

        # Decode image
        image_bytes = base64.b64decode(frame_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # MediaPipe detection (uses module-level detector — created once)
        _, results = mediapipe_detection(frame, detector)
        keypoints = extract_keypoints(results)

        hand_detected = bool(np.any(keypoints != 0))
        sequence.append(keypoints)

        logger.info(f"hand={hand_detected}  seq={len(sequence)}/{SEQUENCE_LENGTH}")

        # ── Warmup: need 30 frames before ML inference ────────────────────────
        if len(sequence) < SEQUENCE_LENGTH:
            return jsonify({
                'success':            True,
                'requiresMoreFrames': True,
                'progress':           len(sequence),
                'total':              SEQUENCE_LENGTH,
                'handDetected':       hand_detected,
                'message':            f'Collecting {len(sequence)}/{SEQUENCE_LENGTH}'
            }), 200

        # ── Step 1: Heuristic Detection (HELLO, YES, STOP, etc.) ─────────────
        h_sign, h_conf = detect_sign_heuristic(keypoints)
        h_history.append(h_sign)

        valid = [s for s in h_history if s is not None]
        if valid:
            best = max(set(valid), key=valid.count)
            if h_history.count(best) >= 3:
                display_word = word_map.get(best, best)
                return jsonify({
                    'success':            True,
                    'requiresMoreFrames': False,
                    'predictedSign':      best,
                    'mappedWord':         display_word,
                    'confidence':         float(h_conf),
                    'handDetected':       hand_detected,
                    'message':            'OK (Heuristic)'
                }), 200

        # ── Step 2: ML Model Detection (A–F) ─────────────────────────────────
        input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)

        # Use direct call — much faster than model.predict() on CPU
        res     = model(input_data, training=False).numpy()[0]
        max_idx = int(np.argmax(res))
        conf    = float(res[max_idx])

        logger.info(f"ML → {actions[max_idx]}  conf={conf:.2f}")

        predictions.append(max_idx)
        predictions = predictions[-10:]

        if conf > THRESHOLD and predictions.count(max_idx) >= 4:
            predicted_action = str(actions[max_idx])
            return jsonify({
                'success':            True,
                'requiresMoreFrames': False,
                'predictedSign':      predicted_action,
                'mappedWord':         predicted_action,
                'confidence':         conf,
                'handDetected':       hand_detected,
                'message':            'OK (ML)'
            }), 200

        # ── Step 3: Still detecting ───────────────────────────────────────────
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
    global sequence, predictions, h_history
    sequence.clear()
    predictions = []
    h_history.clear()
    return jsonify({'success': True, 'message': 'Reset OK'}), 200


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Flask starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=False)