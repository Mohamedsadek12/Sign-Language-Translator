import os, json
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ==============================================================================
# CONFIG
# ==============================================================================
train_data = "dataset/asl_alphabet_train"
MODEL_TASK_PATH = "hand_landmarker.task"   # download once, see instructions below
OUT_MODEL_PATH = "asl_landmark_model.h5"
OUT_CLASSES_PATH = "landmark_classes.json"

# ==============================================================================
# STEP 0 — download the MediaPipe hand landmark model (one-time)
# ==============================================================================
# Run this once from a terminal (network egress required):
#   wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# or in Python:
# if not os.path.exists(MODEL_TASK_PATH):
#     import urllib.request
#     url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
#     print(f"Downloading {MODEL_TASK_PATH} ...")
#     urllib.request.urlretrieve(url, MODEL_TASK_PATH)
#     print("Done")

# ==============================================================================
# STEP 1 — set up the landmarker (IMAGE mode: one-off detections on static files)
# ==============================================================================
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_TASK_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
landmarker = HandLandmarker.create_from_options(options)

def extract_landmarks(img_path):
    image = mp.Image.create_from_file(img_path)
    result = landmarker.detect(image)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]  # first detected hand
    coords = np.array([[p.x, p.y] for p in lm])  # (21, 2)

    # normalize: translate wrist to origin, scale by hand size
    wrist = coords[0].copy()
    coords -= wrist
    scale = np.linalg.norm(coords[9])  # middle finger MCP, post-translation
    if scale < 1e-6:
        return None
    coords /= scale
    return coords.flatten()  # (42,)

# ==============================================================================
# STEP 2 — walk the dataset, extract landmarks for every image
# ==============================================================================
X, y = [], []
skipped = 0

# "nothing" has no hand in frame by design — a landmark vector can't represent
# "no hand", so this class can't exist in this pipeline. The webcam script
# already handles the no-hand case natively (see the "No hand detected" branch).
EXCLUDED_CLASSES = {"nothing"}

class_names = sorted(os.listdir(train_data))
for class_name in class_names:
    class_dir = os.path.join(train_data, class_name)
    if not os.path.isdir(class_dir):
        continue
    if class_name in EXCLUDED_CLASSES:
        print(f"Skipping {class_name} (excluded — no hand in frame by design)")
        continue
    files = os.listdir(class_dir)
    print(f"Processing {class_name}: {len(files)} images")
    for fname in files:
        vec = extract_landmarks(os.path.join(class_dir, fname))
        if vec is not None:
            X.append(vec)
            y.append(class_name)
        else:
            skipped += 1

landmarker.close()

X = np.array(X)
print(f"\nTotal usable samples: {len(X)}")
print(f"Skipped (no hand detected): {skipped}")

# ==============================================================================
# STEP 3 — encode labels, split, train
# ==============================================================================
from collections import Counter
counts = Counter(y)
too_few = [cls for cls, c in counts.items() if c < 2]
if too_few:
    print(f"Dropping classes with <2 usable samples: {too_few}")
    keep_mask = np.array([label not in too_few for label in y])
    X = X[keep_mask]
    y = [label for label in y if label not in too_few]

le = LabelEncoder()
y_int = le.fit_transform(y)
y_cat = to_categorical(y_int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_int
)

model = Sequential([
    Dense(128, activation='relu', input_shape=(42,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_cat.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

val_acc = max(history.history['val_accuracy'])
print(f"\nBest val accuracy: {val_acc:.2%}")

# ==============================================================================
# STEP 4 — save model + class index mapping (index -> letter, matches argmax output)
# ==============================================================================
model.save(OUT_MODEL_PATH)
with open(OUT_CLASSES_PATH, 'w') as f:
    json.dump({int(i): cls for i, cls in enumerate(le.classes_)}, f)

print(f"Saved model to {OUT_MODEL_PATH}")
print(f"Saved class mapping to {OUT_CLASSES_PATH}")