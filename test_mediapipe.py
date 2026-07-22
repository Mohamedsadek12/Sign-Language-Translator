import cv2
import numpy as np
import json
import time
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from tensorflow.keras.models import load_model

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_PATH      = 'asl_landmark_model.h5'
CLASSES_PATH    = 'landmark_classes.json'
HAND_TASK_PATH  = 'hand_landmarker.task'   # same file used in training
CONF_THRESHOLD  = 60.0

# ==============================================================================
# LOAD CLASSIFIER + LABELS
# ==============================================================================
model = load_model(MODEL_PATH)
print("Model loaded")

with open(CLASSES_PATH, 'r') as f:
    idx_to_class = json.load(f)
print("Class labels loaded")

# ==============================================================================
# SET UP HAND LANDMARKER — VIDEO mode (frame-by-frame, uses tracking internally)
# ==============================================================================
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_TASK_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
landmarker = HandLandmarker.create_from_options(options)

# ==============================================================================
# NORMALIZATION — must exactly match training
# ==============================================================================
def normalize_landmarks(hand_landmarks):
    coords = np.array([[p.x, p.y] for p in hand_landmarks])  # (21, 2)
    wrist = coords[0].copy()
    coords -= wrist
    scale = np.linalg.norm(coords[9])
    if scale < 1e-6:
        return None
    coords /= scale
    return coords.flatten()

# simple skeleton connections for drawing (21-point hand model)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17)              # palm
]

# ==============================================================================
# WEBCAM LOOP
# ==============================================================================
cap = cv2.VideoCapture(0)
print("Webcam started — press Q to quit")
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    label, confidence = None, 0.0
    h, w = frame.shape[:2]

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]  # list of NormalizedLandmark

        # draw skeleton
        pts = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)

        vec = normalize_landmarks(hand_landmarks)
        if vec is not None:
            pred = model.predict(np.expand_dims(vec, axis=0), verbose=0)
            idx = int(np.argmax(pred))
            label = idx_to_class[str(idx)]
            confidence = float(np.max(pred) * 100)

    # ==============================================================================
    # DISPLAY
    # ==============================================================================
    if label is not None:
        text = f"{label} ({confidence:.1f}%)" if confidence >= CONF_THRESHOLD else "..."
        color = (0, 255, 0) if confidence >= CONF_THRESHOLD else (0, 165, 255)

        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

        bar_width = int((confidence / 100) * 300)
        cv2.rectangle(frame, (20, 65), (20 + bar_width, 85), color, -1)
        cv2.rectangle(frame, (20, 65), (320, 85), (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No hand detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow('ASL Translator (landmarks) — Press Q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()