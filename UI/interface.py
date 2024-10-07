import gradio as gr
import cv2
import numpy as np

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