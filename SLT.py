import os
import json
import matplotlib.pyplot as plt
import cv2
 
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ======================================================================================================

train_data = "dataset/asl_alphabet_train"
IMG_SIZE   = 64
BATCH_SIZE = 32
SEED       = 42 # to give the same split each time, same as random_state in train_test_split
EPOCHS     = 25

classes = sorted([
    d for d in os.listdir(train_data)
    if os.path.isdir(os.path.join(train_data, d))
])

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
NUM_CLASSES = train_generator.num_classes

print("Class indices mapping:", train_generator.class_indices)
print(f"Train samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# construct class for custom cnn layer to be inferred whenever
class CustomDenseLayer(keras.layers.Layer):
    def __init__(self, units=32, activation='relu', **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.activation_fn = tf.keras.activations.get(activation)
        # inheriting and overriding keras' dense layer
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(self.units,),initializer='zeros', trainable=True)
    def call(self, inputs):
        z = tf.matmul(inputs, self.w) + self.b
        if self.activation_fn is not None:
            return self.activation_fn(z)
        return z
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'activation': self.activation_name})
        return config

# Model Building

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Classification head
    Flatten(),
    CustomDenseLayer(256, activation='relu'),
    Dropout(0.5),                                      # heavy dropout before final layer
    CustomDenseLayer(NUM_CLASSES, activation=None),    # raw logits — no activation before Softmax
    Softmax()
])

# model compile and build

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy'])

print("model summary before building:")
model.summary()

# Callbacks
callbacks = [
    # Stop early if val_loss stops improving
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Halve the learning rate when val_loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    # Save only the best model checkpoint based on val_accuracy
    ModelCheckpoint(
        filepath='best_asl_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

loss, accuracy = model.evaluate(val_generator, verbose=1)
print(f'Validation loss: {loss:.4f}')
print(f'Validation accuracy: {accuracy:.4f}')


# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()


# Save the model and class indices mapping
model.save('asl_custom_cnn.keras')
print("Model saved as successfully.")

# Flip the mapping: {0: 'A', 1: 'B', ...} so predictions are human-readable
idx_to_class = {str(v): k for k, v in train_generator.class_indices.items()}
with open('class_indices.json', 'w') as f:
    json.dump(idx_to_class, f, indent=2)
print("Class indices saved → class_indices.json")
print(idx_to_class)