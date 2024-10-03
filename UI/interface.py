import gradio as gr
import cv2
import numpy as np
# temporary 
from fer import FER

import importlib.util
import sys
import os

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MMNet-main'))

# Add the MMNet-main directory to sys.path
sys.path.append(module_path)

# Define the module name and path to the function
module_name = "4DME"
function_name = "run_training"
file_path = os.path.join(module_path, f"{module_name}.py")

# Load the module dynamically
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Now you can access run_training() from the module
run_training = getattr(module, function_name)


# testing with built in emotion detector
emotion_detector = FER(mtcnn=True)

# def detect_emotion(image):
#     # convert image to BGR (OpenCV format)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
#     # emotion detection
#     emotions = emotion_detector.detect_emotions(image)
    
#     if emotions:
#         dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
#         # display box around the face
#         (x, y, w, h) = emotions[0]['box']
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # text with emotion
#         cv2.putText(image, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
#     # converts back to RGB for display
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     return image, dominant_emotion if emotions else "No face detected"

def detect_emotion(excel):
    return run_training(excel)


# Gradio interface
iface = gr.Interface(
    fn = detect_emotion,
    inputs = gr.File(),
    outputs = [gr.Image(), gr.Text(label="Detected Micro-Expression")],
    title = "Facial Emotion Recognition",
    description = "Upload an image to detect the micro-expression."
)

iface.launch()