import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("resnet_model.h5")

# Preprocessing function
def preprocess_image(image_pil):
    image = np.array(image_pil.convert("L"))  # Convert to grayscale
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = tf.keras.applications.resnet.preprocess_input(image.astype(np.float32))
    return np.expand_dims(image, axis=0), image

# Grad-CAM generation
def generate_gradcam(model, input_array, original_image, layer_name='conv5_block3_out'):
    base_model = model.layers[0]  # Extract inner ResNet50
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(layer_name).output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_array)
        class_output = predictions[:, 0]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap + 1e-8)

    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Blend with original image using alpha=0.4 for heatmap
    overlay = cv2.addWeighted(original_image, 0.6, heatmap_color, 0.4, 0)

    return overlay

# Full classification function
def classify(image):
    input_array, display_img = preprocess_image(image)
    prediction = model.predict(input_array)[0][0]
    label = "🩸 Cancer" if prediction > 0.5 else "✅ Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    overlay = generate_gradcam(model, input_array, display_img)
    return f"{label} ({confidence:.2%})", overlay

# Gradio app setup
demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs=["text", "image"],
    title="🎀 Hope in Pixels: Breast Cancer CT Classifier",
    description="Upload a grayscale CT scan and get a cancer prediction with a clear Grad-CAM heatmap."
)

if __name__ == "__main__":
    demo.launch()
