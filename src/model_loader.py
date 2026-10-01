# Loading the trained CNN model from native keras file.
# Keeps entire model, gives full named-layer access which is what Grad-CAM needs.
# Note: requires TensorFlow 2.17.0 on Python 3.11/12 to match environment in model training.
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import streamlit as st
import tensorflow as tf
import keras
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_DIR / 'model' / "idc_cnn_final.h5"

GRAD_CAM_LAYER_NAME = "relu4b" # Can also try conv4b

# Deserialising the model  with custom objects from cnn training

@keras.saving.register_keras_serializable()
class CastToFloat32(keras.layers.Layer):
    def call(self, x):
        return tf.cast(x, "float32")

    def compute_output_spec(self, x):
        return keras.KerasTensor(shape=x.shape, dtype="float32")

@keras.saving.register_keras_serializable()
class CastToFloat16(keras.layers.Layer):
    def call(self, x):
        return tf.cast(x, "float16")

    def compute_output_spec(self, x):
        return keras.KerasTensor(shape=x.shape, dtype="float16")


def _load_model_uncached(model_path: str = str(MODEL_PATH)) -> tf.keras.Model:
    """
    Model loader without caching. Used when script is run.
    """
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "CastToFloat32": CastToFloat32,
            "CastToFloat16": CastToFloat16,
        }
    )

@st.cache_resource
def load_model(model_path: str = str(MODEL_PATH)) -> tf.keras.Model:
    """
    Model loader which caches the model so Streamlit only loads once per session.
    """
    return _load_model_uncached(model_path)

def verify_model(model: tf.keras.Model) -> None:
    """For sanity check below"""
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
    print("\nLayers:")
    for layer in model.layers:
        print(f"  {layer.name:30s} {layer.__class__.__name__}")

# Sanity check for inspection purposes
if __name__ == "__main__":
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    verify_model(m)