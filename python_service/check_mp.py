import mediapipe as mp
import traceback

print("Imported mp")
try:
    mp_hands = mp.solutions.hands
    print("Accessed mp.solutions.hands")
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        print("Hands initialized")
except Exception:
    traceback.print_exc()
