import math
from PIL import Image
import numpy as np
import torchvision.models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os, torch
import torch.nn as nn
#import image_utils
import argparse, random
from functools import partial

from CA_block import resnet18_pos_attention    # originaly imported resnet50 fron ca_block.py, but its not used anywhere so i deleted it, (changes made)

from PC_module import VisionTransformer_POS

from torchvision.transforms import Resize
torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)

class MMNet(nn.Module):
    def __init__(self):
        super(MMNet, self).__init__()


        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=90*2, kernel_size=3, stride=2,padding=1, bias=False,groups=1), # groups variable originally set to 2, set to either 1 or 3 to make it run, (changes made)
            nn.BatchNorm2d(180),
            nn.ReLU(inplace=True),
            )
        self.pos =nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            )
        ##Position Calibration Module(subbranch)
        self.vit_pos=VisionTransformer_POS(img_size=14,
        patch_size=1, embed_dim=512, depth=3, num_heads=4, mlp_ratio=2, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.3)
        self.resize=Resize([14,14])
        ##main branch consisting of CA blocks
        self.main_branch =resnet18_pos_attention()
        self.head1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1 * 112 *112, 38,bias=False),

        )

        self.timeembed = nn.Parameter(torch.zeros(1, 4, 111, 111))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, if_shuffle):
    def forward(self, x1, x5, if_shuffle):
        ##onset:x1 apex:x5
        B = x1.shape[0]

        #Position Calibration Module (subbranch)
        POS =self.vit_pos(self.resize(x1)).transpose(1,2).view(B,512,14,14)
        act = x5 -x1
        act=self.conv_act(act)
        #main branch and fusion
        out,_=self.main_branch(act,POS)

        return out

# Load the pre-trained model from file
def load_model(model_path):
    model = MMNet()  # Initialize your MMNet model
    model.load_state_dict(torch.load(model_path))  # Load pre-trained model weights
    model.eval()  # Set model to evaluation mode
    return model

# Preprocess the input image for prediction
def preprocess_image(image):
    print(f"Input type: {type(image)}")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # If input is a string, treat it as a file path and open the image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    # If input is a NumPy array (from Gradio), convert it to PIL Image
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    # If input is a PIL Image, make sure it's in RGB mode
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise ValueError("Input should be an image path, a NumPy array, or a PIL image object.")
    
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Function to predict emotion using onset and apex frames
def predict_emotion(model, onset_image_path, apex_image_path):
    # Preprocess onset and apex frames
    onset_frame = preprocess_image(onset_image_path)  # (1, 3, 224, 224)
    apex_frame = preprocess_image(apex_image_path)    # (1, 3, 224, 224)
    
    # Move to the appropriate device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    onset_frame = onset_frame.to(device)
    apex_frame = apex_frame.to(device)
    
    # Forward pass through the model
    with torch.no_grad():
        output = model(onset_frame, apex_frame, False)  # Modify this as per your model structure

    # Get the predicted class (emotion) using torch.max
    _, predicted_class = torch.max(output, 1)  # Assuming model output is logits or probabilities
    return predicted_class.item()

def main():
    # Path to the pre-trained model weights
    model_path = 'model_weights_subject_1.pth'  # Replace with your model path

    # Paths to the onset and apex frames
    onset_image_path = r"C:\Users\jeret\OneDrive\Documents\GitHub\Final-Year-Project\MMNet-main\datasets\CASME2\Cropped-updated\Cropped\1\EP02_01f\reg_img46.jpg"  # Replace with actual onset image path
    apex_image_path = r"C:\Users\jeret\OneDrive\Documents\GitHub\Final-Year-Project\MMNet-main\datasets\CASME2\Cropped-updated\Cropped\1\EP02_01f\reg_img59.jpg"    # Replace with actual apex image path

    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist. Please provide a valid path.")
    else:
        # Load the pre-trained model
        model = load_model(model_path)

        # Predict the emotion from onset and apex frames
        predicted_emotion = predict_emotion(model, onset_image_path, apex_image_path)

        # Print the predicted emotion
        print(f"Predicted Emotion Class: {predicted_emotion}")


# Main entry point for the script
if __name__ == "__main__":
    main()
