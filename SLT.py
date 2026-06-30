import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
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


# Model Building

model = Sequential([


])