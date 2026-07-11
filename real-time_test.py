import cv2
import numpy as np
import tensorflow as tf
import keras
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Softmax

# ==============================================================================
# CUSTOM LAYER
# ==============================================================================
class CustomDenseLayer(keras.layers.Layer):
    def __init__(self, units=32, activation='relu', **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation_fn = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        if self.activation_fn is not None:
            return self.activation_fn(z)
        return z

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'activation': self.activation_name})
        return config

# ==============================================================================
# LOAD MODEL + CLASS INDICES
# ==============================================================================
IMG_SIZE = 64

model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)), Dropout(0.25),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)), Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)), Dropout(0.25),

    Flatten(),
    CustomDenseLayer(256, activation='relu'),
    Dropout(0.5),
    CustomDenseLayer(29, activation=None),
    Softmax()
])

model.load_weights('asl_custom_cnn.h5')

with open('class_indices.json', 'r') as f:
    idx_to_class = json.load(f)

print("✅ Model loaded — press Q to quit")

# ==============================================================================
# WEBCAM LOOP
# ==============================================================================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest — box in the center where you put your hand
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = w//2 - 150, h//2 - 150, w//2 + 150, h//2 + 150
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI and predict
    img_input  = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_input  = np.expand_dims(img_input, axis=0)
    pred       = model.predict(img_input, verbose=0)
    label      = idx_to_class[str(np.argmax(pred))]
    confidence = np.max(pred) * 100

    # Draw box and prediction on frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} ({confidence:.1f}%)",
                (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3)

    cv2.imshow('ASL Translator — Press Q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()