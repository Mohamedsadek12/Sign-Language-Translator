import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==============================================================================
# CONFIG
# ==============================================================================
IMG_SIZE   = 128    # must match what you trained with
MODEL_PATH = 'best_asl_mobilenet.h5'
INDICES_PATH = 'class_indices.json'

# ==============================================================================
# BUILD MODEL + LOAD WEIGHTS
# ==============================================================================
base_model = MobileNetV2(
    input_shape = (IMG_SIZE, IMG_SIZE, 3),
    include_top = False,
    weights     = None   # no imagenet weights needed — we load our own
)

x      = base_model.output
x      = GlobalAveragePooling2D()(x)
x      = BatchNormalization()(x)
x      = Dense(256, activation='relu')(x)
x      = Dropout(0.5)(x)
x      = Dense(128, activation='relu')(x)
x      = Dropout(0.3)(x)
output = Dense(29, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.load_weights(MODEL_PATH)
print("✅ Model loaded")

# ==============================================================================
# LOAD CLASS INDICES
# ==============================================================================
with open(INDICES_PATH, 'r') as f:
    idx_to_class = json.load(f)
print("✅ Class indices loaded")

# ==============================================================================
# WEBCAM LOOP
# ==============================================================================
cap = cv2.VideoCapture(0)
print("📷 Webcam started — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Define ROI box in center of frame
    h, w   = frame.shape[:2]
    x1, y1 = w//2 - 150, h//2 - 150
    x2, y2 = w//2 + 150, h//2 + 150
    roi    = frame[y1:y2, x1:x2]

    # Preprocess — MobileNetV2 needs preprocess_input NOT /255
    img_rgb   = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_input   = preprocess_input(img_resized.astype(np.float32))
    img_batch   = np.expand_dims(img_input, axis=0)

    # Predict
    pred       = model.predict(img_batch, verbose=0)
    label      = idx_to_class[str(np.argmax(pred))]
    confidence = np.max(pred) * 100

    # Draw ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw prediction above box
    cv2.putText(frame, f"{label} ({confidence:.1f}%)",
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3)

    # Show confidence bar
    bar_width = int((confidence / 100) * 300)
    cv2.rectangle(frame, (x1, y2 + 10), (x1 + bar_width, y2 + 30),
                  (0, 255, 0), -1)
    cv2.rectangle(frame, (x1, y2 + 10), (x2, y2 + 30),
                  (255, 255, 255), 2)

    cv2.imshow('ASL Translator — Press Q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()