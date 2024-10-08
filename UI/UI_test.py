import gradio as gr
import cv2
import numpy as np
import torch
import importlib.util
import sys
import os
from PIL import Image

# Dynamically import the predict_emotion module
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MMNet-main'))
sys.path.append(module_path)

module_name = "predict_emotion"
file_path = os.path.join(module_path, f"{module_name}.py")

# Load the predict_emotion module dynamically
spec = importlib.util.spec_from_file_location(module_name, file_path)
emotion_predict_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = emotion_predict_module
spec.loader.exec_module(emotion_predict_module)

# Now you can access predict_emotion and preprocess_image from predict_emotion module
predict_emotion = getattr(emotion_predict_module, "predict_emotion")
load_model = getattr(emotion_predict_module, "load_model")
preprocess_image = getattr(emotion_predict_module, "preprocess_image")

# Detect emotion function
def detect_emotion(onset_image, apex_image):
    # Debug: Print the input types
    print(f"Onset Image Type: {type(onset_image)}")
    print(f"Apex Image Type: {type(apex_image)}")

    # Assuming that Gradio will always provide a PIL image if type is set to 'pil'
    # onset_frame = preprocess_image(onset_image)  # No need for extra checks
    # apex_frame = preprocess_image(apex_image)    # No need for extra checks

    # Predict emotion using onset and apex frames
    predicted_emotion = predict_emotion(model, onset_image, onset_image)
    if predicted_emotion == 0:
        predicted_emotion = 'Positive'
    elif predicted_emotion == 1:
        predicted_emotion = 'Surprise'
    else:
        predicted_emotion = 'Others'

    return f"Predicted Emotion: {predicted_emotion}"

# Main entry point
if __name__ == "__main__":
    # Path to your model weights
    model_path = os.path.join(module_path, "model_weights.pth")
    model = load_model(model_path)  # Load the pre-trained model

    iface = gr.Interface(
        fn=detect_emotion,
        inputs=[
            gr.Image(),  # Onset image input
            gr.Image()  # Apex image input
        ],
        outputs=[gr.Text(label="Detected Micro-Expression")],
        title="Facial Emotion Recognition",
        description="Upload onset and apex images along with a CSV file to detect the micro-expression."
    )


    # Launch Gradio app
    iface.launch()
