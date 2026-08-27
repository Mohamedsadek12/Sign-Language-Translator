import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Config
IMG_SIZE   = 128 # must match what trained with
MODEL_PATH = 'best_asl_mobilenet.h5'
INDICES_PATH = 'class_indices.json'

# Build Moldel + Load Weights
base_model = MobileNetV2(
    input_shape = (IMG_SIZE, IMG_SIZE, 3),
    include_top = False,
    weights     = None   # no imagenet weights needed we load our own
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(29, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.load_weights(MODEL_PATH)
print("Model loaded successfully")

# Load Class Idices
with open(INDICES_PATH, 'r') as f:
    idx_to_class = json.load(f)
print("Class indices loaded successfully")


# Webcam Loop
cap = cv2.VideoCapture(0)
print("Webcam started — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Do NOT flip before this point, the model was trained with
    # horizontal_flip=False, so it needs the frame in its original,
    # unmirrored orientation to match training data.
    h, w   = frame.shape[:2]
    x1, y1 = w//2 - 150, h//2 - 150
    x2, y2 = w//2 + 150, h//2 + 150
    roi    = frame[y1:y2, x1:x2]

    img_rgb     = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_input   = preprocess_input(img_resized.astype(np.float32))
    img_batch   = np.expand_dims(img_input, axis=0)

    pred       = model.predict(img_batch, verbose=0)
    label      = idx_to_class[str(np.argmax(pred))]
    confidence = np.max(pred) * 100

    # NOW flip, purely for a natural mirror-view display.
    # Prediction already happened above on the correct orientation.
    frame = cv2.flip(frame, 1)

    # Because we flipped after cropping, the ROI box position must be
    # mirrored too, or the green box will be drawn on the wrong side
    # of the now-flipped frame.
    x1_disp, x2_disp = w - x2, w - x1

    cv2.rectangle(frame, (x1_disp, y1), (x2_disp, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} ({confidence:.1f}%)",
                (x1_disp, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3)

    bar_width = int((confidence / 100) * 300)
    cv2.rectangle(frame, (x1_disp, y2 + 10), (x1_disp + bar_width, y2 + 30),
                  (0, 255, 0), -1)
    cv2.rectangle(frame, (x1_disp, y2 + 10), (x2_disp, y2 + 30),
                  (255, 255, 255), 2)

    cv2.imshow('ASL Translator — Press Q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()