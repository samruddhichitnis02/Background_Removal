import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from train import create_dir
import gdown
import streamlit as st


# URL of the model stored in Google Drive
file_id = '1Wv8KsXQlk1pRgu-r9OYGWeEbtWdaaAeA'
url = f'https://drive.google.com/uc?id={file_id}'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Global Parameters 
image_h = 256
image_w = 256


@st.cache_resource
def load_model_from_drive(url):
    output = 'model.h5'
    gdown.download(url, output, quiet = False)
    model = tf.keras.models.load_model(output)
    return model



def prediction(img):
    """
    Calls the model and removes the background for the given image

    Args:
        img: Image
    
    Returns:
        Image with background removed
    """
    # Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    # Directory for storing the mask

    # Loading the model
    model = load_model_from_drive(url)
    # model = tf.keras.models.load_model("/workspaces/Background_Removal/model.h5")

    h, w, _ = img.shape
    x = cv2.resize(img, (image_w, image_h))
    x = x/255.0
    x = x.astype(np.float32) # (256, 256, 3)
    x = np.expand_dims(x, axis = 0) #(1, 256, 256, 3)

    # Prediction
    y = model.predict(x, verbose=0)[0][:,:,-1]

    # reshaping the predicted image to the original image shape
    y = cv2.resize(y, (w, h)) # shape of the image - w, h
    y = np.expand_dims(y, axis=-1) # shape of the image - w, h, 1

    # Save the image 
    # mask gives keeps the foreground pixels white and all the background pixels are black
    # hence when image and mask are multiplied, all the background pixels remove the background from the 
    # original image.
    masked_image = img * y
    
    line = np.ones((h, 10, 3)) * 128
    cat_images = np.concatenate([img, line, masked_image], axis=1)
    return cat_images