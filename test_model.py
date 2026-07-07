import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras
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
# CONFIG
# ==============================================================================
IMG_SIZE = 64
TEST_DIR = 'dataset/asl_alphabet_test'

# ==============================================================================
# REBUILD ARCHITECTURE + LOAD WEIGHTS
# ==============================================================================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    CustomDenseLayer(256, activation='relu'),
    Dropout(0.5),
    CustomDenseLayer(29, activation=None),
    Softmax()
])

model.load_weights('asl_custom_cnn.h5')
print("✅ Model loaded successfully")

# ==============================================================================
# LOAD CLASS INDICES
# ==============================================================================
with open('class_indices.json', 'r') as f:
    idx_to_class = json.load(f)
print("✅ Class indices loaded")

# ==============================================================================
# TEST IMAGES
# ==============================================================================
test_images = [f for f in os.listdir(TEST_DIR)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(test_images)} images\n")

fig, axes = plt.subplots(1, len(test_images), figsize=(4 * len(test_images), 4))
if len(test_images) == 1:
    axes = [axes]

for ax, img_name in zip(axes, test_images):
    img_path  = os.path.join(TEST_DIR, img_name)
    img       = cv2.imread(img_path)
    img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)) / 255.0, axis=0)

    pred       = model.predict(img_input, verbose=0)
    label      = idx_to_class[str(np.argmax(pred))]
    confidence = np.max(pred) * 100

    ax.imshow(img_rgb)
    ax.set_title(f"{label}\n{confidence:.1f}%", fontsize=12)
    ax.axis('off')
    print(f"{img_name} → Predicted: {label} ({confidence:.1f}%)")

plt.tight_layout()
plt.savefig('test_results.png', dpi=150)
plt.show()