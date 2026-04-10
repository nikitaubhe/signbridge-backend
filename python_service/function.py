# import os
# os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
# os.environ["XDG_RUNTIME_DIR"] = "/tmp"
 
# import cv2
# import numpy as np
# import mediapipe as mp
 
# actions = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
# no_sequences = 30
# sequence_length = 30
 
# _detector = None
 
# def get_detector():
#     global _detector
#     if _detector is None:
#         _detector = mp.solutions.hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#     return _detector
 
# def mediapipe_detection(image, detector_instance):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_rgb.flags.writeable = False
#     results = detector_instance.process(image_rgb)
#     image_rgb.flags.writeable = True
#     image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
#     return image, results
 
# def extract_keypoints(results):
#     if results.multi_hand_landmarks:
#         hand = results.multi_hand_landmarks[0]
#         rh = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
#         return rh
#     return np.zeros(21 * 3)
 
# def draw_landmarks_on_image(rgb_image, results):
#     annotated_image = np.copy(rgb_image)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             for landmark in hand_landmarks.landmark:
#                 h, w, _ = annotated_image.shape
#                 x, y = int(landmark.x * w), int(landmark.y * h)
#                 cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)
#     return annotated_image
 


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





import cv2
import numpy as np
import os
import mediapipe as mp

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
no_sequences = 30
sequence_length = 30

# ── Old API (works without OpenGL) ──────────────────────────────────────────
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def mediapipe_detection(image, detector_instance):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = detector_instance.process(image_rgb)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        rh = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
        return rh
    return np.zeros(21 * 3)

def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Custom drawing utility since mp.solutions.drawing_utils expects normalized landmarks 
    from the old API, but we can adapt or just write a simple drawer.
    Actually, let's stick to simple drawing or adapting.
    """
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    # Use old drawing utils if possible, by converting? 
    # Or just draw manually. Manual is safer as object types differ.
    
    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:
            # Loop through landmarks
            for landmark in hand_landmarks:
                h, w, _ = annotated_image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)
                
    return annotated_image