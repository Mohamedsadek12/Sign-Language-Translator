import os, json, cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

train_data  = "dataset/asl_alphabet_train"
IMG_SIZE = 128 # MobileNetV2 works better with 96+
BATCH_SIZE = 32
SEED = 42
EPOCHS_P1 = 10 # Phase 1 — frozen base
EPOCHS_P2 = 20 # Phase 2 — fine tuning

# Data Generators
train_data_gen = ImageDataGenerator(
    preprocessing_function = preprocess_input, # scales to [-1, 1]
    validation_split = 0.2,
    rotation_range = 15,
    width_shift_range = 0.15,
    height_shift_range = 0.15,
    zoom_range = 0.15,
    brightness_range = [0.6, 1.4],  # more aggressive than before
    shear_range = 0.1, # slight shear for hand tilt
    horizontal_flip = False
)

val_data_gen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    validation_split = 0.2
)

train_generator = train_data_gen.flow_from_directory(
    train_data,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'categorical', # one-hot encoding
    subset = 'training',
    shuffle = True, # shuffle the training data to help the model generalize better and avoid learning the order of the data.
    seed = SEED
)
 
val_generator = val_data_gen.flow_from_directory(
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


#Model building

base_model = MobileNetV2(
    input_shape = (IMG_SIZE, IMG_SIZE, 3),
    include_top = False, # remove ImageNet head
    weights     = 'imagenet' # use pretrained weights
)

# Freeze entire base for Phase 1
base_model.trainable = False
print(f"\nBase model layers : {len(base_model.layers)}")
print(f"Trainable (Phase 1): 0 (base frozen)")
 
# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
 
model = Model(inputs=base_model.input, outputs=output)
 
model.compile(
    optimizer = Adam(learning_rate=0.001),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

model.summary()

# phase 1 callbacks (base frozen)
callbacks_p1 = [
    EarlyStopping(
        monitor='val_loss',
        patience=4, 
        restore_best_weights=True, 
        verbose=1
        ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6, 
        verbose=1
        ),
    ModelCheckpoint(
        'best_asl_mobilenet_p1.h5', 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1
        )
]


history_p1 = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs          = EPOCHS_P1,
    callbacks       = callbacks_p1
)


p1_acc = max(history_p1.history['val_accuracy'])
print(f"\nPhase 1 best val accuracy: {p1_acc:.2f}")


# phase 2 — fine tuning
# unfreeze the base model for fine-tuning

base_model.trainable = True
fine_tune_from = len(base_model.layers) - 30

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

for layer in base_model.layers[fine_tune_from:]:
    layer.trainable = True

print(f"Frozen layers  : {fine_tune_from}")
print(f"Trainable layers: {len(base_model.layers) - fine_tune_from}")

model.compile(
    optimizer = Adam(learning_rate=1e-5),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

callbacks_p2 = [
    EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True, 
        verbose=1
        ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-7, 
        verbose=1
        ),
    ModelCheckpoint(
        'best_asl_mobilenet.h5', 
        monitor='val_accuracy', 
        save_best_only=True,
        verbose=1
    )
]

history_p2 = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs          = EPOCHS_P2,
    callbacks       = callbacks_p2
)

p2_acc = max(history_p2.history['val_accuracy'])
print(f"\nPhase 2 best val accuracy: {p2_acc:.2f}")
print(f"Improvement from fine-tuning: +{(p2_acc - p1_acc):.2f}")


# plot training curves
acc = history_p1.history['accuracy'] + history_p2.history['accuracy']
val_acc = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
loss = history_p1.history['loss'] + history_p2.history['loss']
val_loss = history_p1.history['val_loss'] + history_p2.history['val_loss']
p1_end = len(history_p1.history['accuracy'])
 
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
 
axes[0].plot(acc, label='Train')
axes[0].plot(val_acc, label='Validation')
axes[0].axvline(x=p1_end, color='red', linestyle='--', label='Phase 2 starts')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True)
 
axes[1].plot(loss,     label='Train')
axes[1].plot(val_loss, label='Validation')
axes[1].axvline(x=p1_end, color='red', linestyle='--', label='Phase 2 starts')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True)
 
plt.suptitle(f'MobileNetV2 Training — Final Val Accuracy: {p2_acc:.2%}', fontsize=13)
plt.tight_layout()
plt.savefig('mobilenet_training_curves.png', dpi=150)
plt.show()


# save the final model
model.save('asl_mobilenet_final.h5')
print(f"\nFinal val accuracy: {p2_acc:.2%}")


