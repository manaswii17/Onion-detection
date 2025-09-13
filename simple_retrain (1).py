import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Simple data generator without validation split
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2]
)

# Only training generator - no validation split
train_generator = datagen.flow_from_directory(
    '.',
    target_size=(224, 224),
    batch_size=4,
    classes=['onion_images'],
    class_mode='binary',
    shuffle=True
)

# Model setup
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze more layers for better learning
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training model without validation split...")

# Train without validation
history = model.fit(
    train_generator,
    epochs=30,  # More epochs since no validation
    verbose=1
)

# Save the model
model.save('onion_detector_simple.h5')
print("Model saved as onion_detector_simple.h5")
    
