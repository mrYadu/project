import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('smart_agriculture_model.h5')

# Function to predict the class of an image
def predict_image(image_path):
    img = cv2.imread(image_path)
   