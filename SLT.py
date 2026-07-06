import os

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

import tensorflow as tf
from keras.src.layers.activations import activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Softmax
from tensorflow.keras.optimizers import Adam

# ======================================================================================================

train_data = "dataset/asl_alphabet_train"

classes = os.listdir(train_data)

# count the images
counter = 0
for cls in classes:
    cls_path = os.path.join(train_data, cls)
    n_images = len([n for n in os.listdir(cls_path)
                     if n.lower().endswith(('.jpg', '.jpeg', '.png'))])
    counter += n_images
    print(f"Class '{cls}' has {n_images} images.")
print(f"Total number of images: {counter}")


# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
sample_classes = classes[:10]

for ax, cls in zip(axes.flatten(), sample_classes):
    cls_path = os.path.join(train_data, cls)
    img_name = os.listdir(cls_path)[0]
    img = cv2.imread(os.path.join(cls_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(cls)
    ax.axis('off')

plt.tight_layout()
plt.show()


# Data Preprocessing
IMG_SIZE = 64
BATCH_SIZE = 32
SEED = 42 # to give the same split each time, same as random_state in train_test_split

# read images from disk in batches and apply data augmentation to the training set
train_data_gen = ImageDataGenerator(
    rescale = 1.0/255,
    validation_split = 0.2,
    rotation_range = 10, # making the model more robust and reducing overfitting.
    width_shift_range = 0.1, 
    height_shift_range = 0.1, # helps the model handle hands that are not perfectly centered in frame.
    zoom_range = 0.1, # helps with variation in how close the hand is to the camera.
    brightness_range = [0.8, 1.2], # helps the model generalize across different lighting conditions.
    horizontal_flip = False # flipping a hand sign image left-to-right can turn it into a different (or nonsensical) sign
)

# tells the generator to read images directly from a folder structure,
train_generator = train_data_gen.flow_from_directory(
    train_data,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'categorical', # one-hot encoding
    subset = 'training',
    shuffle = True, # shuffle the training data to help the model generalize better and avoid learning the order of the data.
    seed = SEED
)
val_generator = train_data_gen.flow_from_directory(
    train_data,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    subset = 'validation',
    shuffle = False,
    seed = SEED
)


print("Class indices mapping:", train_generator.class_indices)
print(f"Train samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# construct class for custom cnn layer to be inferred whenever
class CustomeDenseLayer(keras.layers.Layer):
    def __init__(self, units=32, activation='relu'):
        super(CustomeDenseLayer, self).__init__()
        self.units = units
        self.activation_name = activation
        self.activation_fn = tf.keras.activations.get(activation)
        # inheriting and overriding keras' dense layer
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,),initializer='zeros', trainable=True)
    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        if self.activation_fn is not None:
            return self.activation_fn(z)
        return z

# Model Building

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dropout(0.3),

    CustomeDenseLayer(128, activation='relu'),
    CustomeDenseLayer(train_generator.num_classes, activation=None),  # raw logits — no ReLU before softmax
    Softmax()
])

# model compile and build

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("model summary before building:")
model.summary()

model.build((None, IMG_SIZE, IMG_SIZE, 3))
print("\nmodel summary after building:")
model.summary()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

loss, accuracy = model.evaluate(val_generator)
print(f'Validation loss: {loss:.4f}')
print(f'Validation accuracy: {accuracy:.4f}')

model.save('1st_custome_cnn.keras')