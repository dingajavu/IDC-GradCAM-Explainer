import numpy as np
import tensorflow as tf
from PIL import Image

PATCH_SIZE = (50, 50)
CLASS_NAMES = {0: "Benign", 1: "Malignant (IDC)"}

def preprocess_image(image: Image.Image) -> np.ndarray:
    """

    """
    image = image.convert("RGB").resize(PATCH_SIZE)
    arr = np.array(image).astype("float32")
    arr = arr/250.0
    return np.expand_dims(arr, axis=0)

def predict_image(model: tf.keras.Model, image: Image.Image) -> dict:
    """
    Model inference is run on a single image.

    Returns:
        raw sigmoid value
        predicted class (0 or 1)
        label (Benign or Malignant)
        confidence as a probability
        input_array, reused by Grad-CAM
    """
    input_array = preprocess_image(image)
    probability = float(model.predict(input_array, verbose=0)[0][0])
    predicted_class = int(probability >= 0.5)
    confidence = probability if predicted_class == 1 else 1 - probability

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "input_array": input_array,
        "probability": probability,
        "label": CLASS_NAMES[predicted_class],
    }
