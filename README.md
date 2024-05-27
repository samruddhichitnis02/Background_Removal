## Background Removal using Computer Vision with TensorFlow and Keras 

* Table of Contents
    - Introduction
    - Installation
    - Usage
    - Dataset
    - Model Architecture
    - Training
    - Results

* Introduction :
    This project focuses on removing the background from images using deep learning techniques. I have leveraged TensorFlow and Keras to build and train a neural network that can accurately separate foreground objects(humans) from the background in various images.

* Installation :
    To get started, clone this repository and install the required dependencies:
    - git clone https://github.com/yourusername/background-removal.git
    - cd background-removal
    - pip install -r requirements.txt

* Usage :
    - To train the model from scratch you need to run the train.py python file.
    - In the train.py file you can tune the hyperparameters.

* Dataset :
    - The dataset should include pairs of original images and their corresponding masks(ground truth).
    - The images and masks should be properly aligned and named consistently.
    - For training the current model I have used the data from the following [Person Segmentation Dataset](https://www.kaggle.com/datasets/nikhilroxtomar/person-segmentation?resource=download).