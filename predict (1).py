
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2

# Load the trained model
MODEL_PATH = "onion_detector_model.h5"
TEST_FOLDER = "test_images"
OUTPUT_FOLDER = "output_images"
THRESHOLD = 0.5

print("Loading model...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def classify_img(img_path):
    """Return the model's raw score for an image."""

    print(f"\n--- Processing: {img_path} ---")

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess image for MobileNetV2
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array, verbose=0)
        print(f"Raw prediction: {prediction}")

        # This assumes a single-output binary classifier.
        if prediction.size != 1:
            print(
                "Error: Expected one model output. "
                "Check your model's output shape and class setup."
            )
            return None

        score = float(prediction.reshape(-1)[0])
        print(f"Model score: {score:.4f}")
        return score

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


# Check that the test folder exists
if not os.path.isdir(TEST_FOLDER):
    print(f"Error: Folder '{TEST_FOLDER}' was not found.")
    sys.exit(1)

# Find supported image files
file_list = sorted(
    fname for fname in os.listdir(TEST_FOLDER)
    if fname.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    )
)

if not file_list:
    print(f"No image files found in '{TEST_FOLDER}'.")
    sys.exit(1)

print(f"\nFound {len(file_list)} images.")
print("=" * 50)
print("STARTING CLASSIFICATION")
print("=" * 50)

for fname in file_list:
    img_path = os.path.join(TEST_FOLDER, fname)
    score = classify_img(img_path)

    # Skip images that could not be processed
    if score is None:
        continue

    # This assumes higher scores mean onion.
    # Verify the class mapping in your training code.
    if score >= THRESHOLD:
        action = "Onion detected -> Camera ON -> Cutter OFF"
        color = (0, 255, 0)
        status = "PROTECTED"
    else:
        action = "Other plant detected -> Camera ON -> Cutter ON"
        color = (0, 0, 255)
        status = "CUTTING"

    print(f"\nRESULT for {fname}:")
    print(f"  {action}")
    print(f"  Model score: {score:.4f}")
    print(f"  Status: {status}")

    # Read the original image for the result overlay
    img_cv = cv2.imread(img_path)

    if img_cv is None:
        print("Could not load image for overlay.")
        continue

    # Add result information to the image
    cv2.putText(
        img_cv, status, (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA
    )

    cv2.putText(
        img_cv, f"Model score: {score:.2f}", (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
    )

    cv2.putText(
        img_cv, action, (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
    )

    output_path = os.path.join(OUTPUT_FOLDER, fname)

    if cv2.imwrite(output_path, img_cv):
        print(f"Output saved: {output_path}")
    else:
        print(f"Failed to save output: {output_path}")

print("\n" + "=" * 50)
print("CLASSIFICATION COMPLETE!")
print(f"Check the '{OUTPUT_FOLDER}' folder for result images.")
print("=" * 50)
