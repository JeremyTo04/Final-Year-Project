import gradio as gr
import cv2
import numpy as np
# temporary 
from fer import FER

# testing with built in emotion detector
emotion_detector = FER(mtcnn=True)

def detect_emotion(image):
    # convert image to BGR (OpenCV format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # emotion detection
    emotions = emotion_detector.detect_emotions(image)
    
    if emotions:
        dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
        # display box around the face
        (x, y, w, h) = emotions[0]['box']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # text with emotion
        cv2.putText(image, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # converts back to RGB for display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image, dominant_emotion if emotions else "No face detected"

# Gradio interface
iface = gr.Interface(
    fn = detect_emotion,
    inputs = gr.Image(),
    outputs = [gr.Image(), gr.Text(label="Detected Micro-Expression")],
    title = "Facial Emotion Recognition",
    description = "Upload an image to detect the micro-expression."
)

iface.launch()