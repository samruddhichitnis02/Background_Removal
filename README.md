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
    - The Person Segmentation Dataset consists of 5768 images and 5768 corresponding masks.
    - While training the model from scratch, all the images were placed in one folder and all the corresponding masks were placed in the other folder, the file structure looks like the below structure:

            ├── People_Segmentation
            │       ├── images
            │       └── masks

* Model Architecture :
    - The model is based on the U-Net architecture, which is widely used for image sengmentation tasks.

    ### What is Image Segmentation?
        - Image Segmentation is a fundamental task in computer vision that involves partitioning an image into distinct regions or segments to simplify its analysis and interpretation.
        - Each segment typically corresponds to different objetcs or parts of the image.
        - This technique is essential for various applications such as medical imaging, autonomous driving, and object detection, enabling more precise and meaningful analysis by isolating relevant areas within the image.

    ### What is U-Net Architecture?
        - The U-Net architecture is a type of convolutional neural network (CNN) specifically designed for image segmentation tasks.
        - The primary purpose of this architecture was to address the challenge of limited annotated data in the medical field.
        - This network was designed to effectively leverage a smaller amount of data while maintaining speed and accuracy.
        - It consists of a contarcting path (encoder) that captures context and a symmetric expanding path (decoder) that enables precise localization.
        - The contracting path contains encoder layers that capture contextual information and reduce the spatial resolution of the input, while the expansive path contains decoder layers that decode the encoded data and use the information from the contracting path via skip connections to generate a segmentation map.
        - The contracting path in U-Net is responsible for identifying the relevant features in the input image. 
        - The encoder layers perform convolutional operations that reduce the spatial resolution of the feature maps while increasing their depth, thereby capturing increasingly abstract representations of the input.
        - On the other hand, the expansive path works on decoding the encoded data and locating the features while maintaining the spatial resolution of the input.The decoder layers in the expansive path upsample the feature maps, while also performing convolutional operations.
        - The skip connections from the contracting path help to preserve the spatial information lost in the contracting path, which helps the decoder layers to locate the features more accurately.