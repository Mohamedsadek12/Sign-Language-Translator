import cv2
import numpy as np
import json
import time
from collections import deque
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from tensorflow.keras.models import load_model

# Config
MODEL_PATH      = "wlasl_100_model_1.h5"
CLASSES_PATH    = "idx2word.json"
HAND_TASK_PATH  = "hand_landmarker.task"
SEQUENCE_LENGTH = 30
NUM_LANDMARKS   = 63
TOTAL_FEATURES  = 126          # 2 hands x 63
CONF_THRESHOLD  = 60.0
PREDICT_EVERY   = 5            # re-run the model every N frames, not every frame

model = load_model(MODEL_PATH, compile=False)
with open(CLASSES_PATH, "r") as f:
    idx2word = json.load(f)

# num_hands=2, matching training — your letter script used num_hands=1
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_TASK_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)

# Same normalization your training used (wrist-centered, scaled by dist to landmark 9, x/y/z)
def normalize_hand(hand_landmarks):
    coords = np.array([[p.x, p.y, p.z] for p in hand_landmarks], dtype=np.float32)
    coords -= coords[0].copy()
    scale = np.linalg.norm(coords[9])
    if scale < 1e-6:
        return None
    return (coords / scale).flatten()

def extract_frame_features(result):
    left_hand = np.zeros(NUM_LANDMARKS, dtype=np.float32)
    right_hand = np.zeros(NUM_LANDMARKS, dtype=np.float32)
    if not result.hand_landmarks:
        return np.concatenate([left_hand, right_hand])
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        normalized = normalize_hand(hand_landmarks)
        if normalized is None:
            continue
        label = handedness[0].category_name.lower()
        if label == "left":
            left_hand = normalized
        elif label == "right":
            right_hand = normalized
    return np.concatenate([left_hand, right_hand])

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Rolling buffer of the last SEQUENCE_LENGTH frames — this replaces the
# "one frame in, one prediction out" flow from your letter script
buffer = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(0)
print("Webcam started — press Q to quit")
start_time = time.time()

label, confidence = None, 0.0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect on the unflipped frame — same rule as your letter script
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    buffer.append(extract_frame_features(result))
    frame_count += 1

    # Only predict once the buffer is full, and only every few frames —
    # predicting every single frame is wasteful and makes the label flicker
    if len(buffer) == SEQUENCE_LENGTH and frame_count % PREDICT_EVERY == 0:
        seq = np.expand_dims(np.array(buffer, dtype=np.float32), axis=0)  # (1, 30, 126)
        pred = model.predict(seq, verbose=0)
        idx = int(np.argmax(pred))
        label = idx2word[str(idx)]
        confidence = float(np.max(pred) * 100)

    # Flip for display only, same as before
    h, w = frame.shape[:2]
    display = cv2.flip(frame, 1)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            pts = [(int((1.0 - p.x) * w), int(p.y * h)) for p in hand_landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(display, pts[a], pts[b], (0, 255, 0), 2)
            for p in pts:
                cv2.circle(display, p, 4, (0, 0, 255), -1)

    if len(buffer) < SEQUENCE_LENGTH:
        cv2.putText(display, f"buffering... {len(buffer)}/{SEQUENCE_LENGTH}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
    elif label is not None:
        text = f"{label} ({confidence:.1f}%)" if confidence >= CONF_THRESHOLD else "..."
        color = (0, 255, 0) if confidence >= CONF_THRESHOLD else (0, 165, 255)
        cv2.putText(display, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

    cv2.imshow("ASL Word Recognition — Press Q to quit", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()