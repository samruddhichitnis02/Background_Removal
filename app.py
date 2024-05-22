import os
import cv2
import sys
import string
import random
import base64
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO, StringIO
from streamlit.components.v1 import html

from predict import prediction

# set tab title
st.set_page_config(page_title = "Background Removal", layout = "wide")

# Remove the white space 
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

# Heading
st.markdown("<h1 style='text-align: center; color: Black; padding-top: 2px'> Background Removal</h1>", unsafe_allow_html = True)

# Change background image
def set_bg(img):
    ext = 'png'
    page_bg_img = f"""
                <style>
                            [data-testid="stAppViewContainer"] > .main {{
                                            background: url(data:image/{ext};base64,{base64.b64encode(open(img, "rb").read()).decode()});
                                            background-size: cover;
                                            background-position: top left;
                                            background-repeat: no-repeat;
                                            background-attachment: fixed;
                            }}

                            [data-testid="stHeader"]{{
                                            background: rgba(0,0,0,0);
                            }}

                </style>
    """ 
    st.markdown(page_bg_img, unsafe_allow_html = True)

set_bg("background.png")

# Predict for the given image
uploaded_file = st.file_uploader("Upload Files", type = ['png','jpeg', 'jpg'])
result = st.button(label = "Submit")
if uploaded_file is not None and result == True:

    print("Image Uploaded Successfully!")

    # This image is in form of image bytes
    image = uploaded_file.read()

    # Converting image byte to an array
    decoded = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    print("Converted the image from byte to an array successfully!")

    mask = prediction(decoded)
    print("Background Removed Successfully!")

    res = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 9))
    save = res + '.JPG'

    cv2.imwrite(save, mask)
    print("Image saved successfully!")

    image = Image.open(save)
    img = st.image(image, use_column_width = True, clamp = True)
    