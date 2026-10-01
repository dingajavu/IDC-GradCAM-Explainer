import numpy as np
import cv2
import tensorflow as tf

def gradcam_heatmap(
        input_array: np.ndarray,
        model: tf.keras.Model,
        last_conv_layer_name: str,
) -> np.ndarray:
    """"
    Computes the GradCAM heatmap given an input image and a model.

    Returns a (H, W) heatmap normalized to [0, 1], where H, W match the
    relu4b spatial dimensions (will be upscaled to the patch
    size separately, in overlay_heatmap)
    """
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(input_array, training=False)
        loss = predictions[:,0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.cast(heatmap, tf.float32)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4,
) -> np.ndarray:
    """
    Resize the heatmap to match the original image and overlay it.

    original_image: (H, W, 3) numpy array in 0-255 range

    returns: (H, W, 3) array, overlaid image.
    """
    h,w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w,h))
    heatmap_unint8 = np.uint8(255 * heatmap_resized)
    heatmap_colour = cv2.applyColorMap(heatmap_unint8, cv2.COLORMAP_JET)
    heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)

    overlaid =  heatmap_colour * alpha + original_image * (1 - alpha)
    return np.uint8(overlaid)

def summarise_activation_region(heatmap: np.ndarray) -> dict:
    """
    Summarise the activation region using LLM to explain in plain terms
    the intensity and location of the strongest activation region.
    """
    h,w = heatmap.shape
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    vertical = "upper" if peak_y < h / 2 else "lower"
    horizontal = "left" if peak_x < w / 2 else "right"

    intensity = float(heatmap.max())
    if intensity >= 0.75:
        intensity_label= "strongly"
    elif intesity >= 0.4:
        intensity_label= "moderately"
    else:
        intensity_label = "mildly"


    return {
        "peak_location": f"{vertical}-{horizontal}",
        "peak_intensity": intensity,
        "intensity_label": intensity_label,
        "mean_activation": float(heatmap.mean()),
    }