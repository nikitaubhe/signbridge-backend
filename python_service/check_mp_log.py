import sys
with open('mp_log.txt', 'w') as f:
    try:
        import mediapipe as mp
        f.write("Imported mp\n")
        mp_hands = mp.solutions.hands
        f.write("Accessed hands\n")
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
             f.write("Initialized hands\n")
    except Exception as e:
        import traceback
        f.write(traceback.format_exc())
