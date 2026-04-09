# import cv2
# import numpy as np
# import os
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# DATA_PATH = os.path.join('MP_Data') 
# actions = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
# no_sequences = 30
# sequence_length = 30

# # Initialize Global Landmarker (Lazy Loading recommended in main app, but helper here)
# base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
# options = vision.HandLandmarkerOptions(base_options=base_options,
#                                        num_hands=1,
#                                        min_hand_detection_confidence=0.5,
#                                        min_hand_presence_confidence=0.5,
#                                        min_tracking_confidence=0.5)
# detector = vision.HandLandmarker.create_from_options(options)

# def mediapipe_detection(image, detector_instance):
#     """
#     Process image using MediaPipe Tasks API.
#     """
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Convert to MediaPipe Image
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    
#     # Detect
#     detection_result = detector_instance.detect(mp_image)
    
#     # Convert back for display
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, detection_result

# def extract_keypoints(detection_result):
#     """
#     Extracts keypoints from HandLandmarkerResult.
#     """
#     if detection_result.hand_landmarks:
#         # Get first hand
#         hand_landmarks = detection_result.hand_landmarks[0]
#         rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks]).flatten()
#         return rh
#     return np.zeros(21*3)

# def draw_landmarks_on_image(rgb_image, detection_result):
#     """
#     Custom drawing utility since mp.solutions.drawing_utils expects normalized landmarks 
#     from the old API, but we can adapt or just write a simple drawer.
#     Actually, let's stick to simple drawing or adapting.
#     """
#     hand_landmarks_list = detection_result.hand_landmarks
#     annotated_image = np.copy(rgb_image)

#     # Use old drawing utils if possible, by converting? 
#     # Or just draw manually. Manual is safer as object types differ.
    
#     if hand_landmarks_list:
#         for hand_landmarks in hand_landmarks_list:
#             # Loop through landmarks
#             for landmark in hand_landmarks:
#                 h, w, _ = annotated_image.shape
#                 x, y = int(landmark.x * w), int(landmark.y * h)
#                 cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)
                
#     return annotated_image





import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
no_sequences = 30
sequence_length = 30

# =======================

# SAFE MEDIAPIPE SETUP

# =======================

detector = None

try:
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = vision.HandLandmarker.create_from_options(options)

except Exception as e:
    print("MediaPipe not supported:", e)
    detector = None

# =======================

# DETECTION FUNCTION

# =======================

def mediapipe_detection(image, detector_instance):
    if detector_instance is None:
        return image, None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    detection_result = detector_instance.detect(mp_image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, detection_result


# =======================

# KEYPOINT EXTRACTION

# =======================

def extract_keypoints(detection_result):
    if detection_result is None:
        return np.zeros(21 * 3)

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks]).flatten()
        return rh

    return np.zeros(21 * 3)


# =======================

# DRAW LANDMARKS

# =======================

def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    if detection_result is None:
        return annotated_image

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            for landmark in hand_landmarks:
                h, w, _ = annotated_image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)

    return annotated_image

