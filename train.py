import os
import cv2
# import scipy.io
import numpy as np
import pandas as pd

from glob import glob

import tensorflow as tf
from tensorflow import keras
from model import build_model
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
global image_h
global image_w


def create_dir(path):
    """
    Creates folder when called
    
    Args:
        path (str) : Location where the folder needs to be created. 
    
    Returns:
        This function does not return anything but creates a folder in the path/location specified.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_dataset(path, split = 0.1):
    """
    Loads the dataset and splits the dataset in to train and validation set.
    
    Args:
        path (str) : Location where all the images are stored.
        split (float) : splitting ration of the dataset in to training set and validation set.

    Returns:
        This function returns the training dataset and validation dataset.
    """
    #Loading the images and masks
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))

    # Splitting the data into training and testing
    split_size = int(len(X) * split)

    train_x, valid_x = train_test_split(X, test_size = split_size, random_state = 42)
    train_y, valid_y = train_test_split(Y, test_size = split_size, random_state = 42)

    return (train_x, train_y), (valid_x, valid_y)


def read_image(path):
    """
    Reads the image, normalizes the image & returns the image

    Args:
        path (str) : Location of the image 
    
    Returns:
        Image
    """
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_w, image_h))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    """
    Reads the mask 

    Args:
        path (str): Location of the mask

    Returns:
        Mask
    """
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (image_w, image_h))
    x = x.astype(np.float32)                            ## (h, w)
    x = np.expand_dims(x, axis = -1)                    ## (h, w, 1)
    x = np.concatenate([x, x, x, x], axis = -1)         ## (h, w, 4) we have 4 masks
    return x


def tf_parse(x, y):
    """
    As both the functions: read_image and read_mask uses OpenCV and Numpy. So, we cannot directly use them in TensorFlow.
    tf_parse will help in using the read_mask and read_image functions in TensorFlow.

    Args:
        x (list): List containing paths of the images
        y (list): List containing paths of the masks

    Returns:
        x : image tensor
        y : mask tensor
    """
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([image_h, image_w, 3])
    y.set_shape([image_h, image_w, 4])
    return x, y


def tf_dataset(X, Y, batch = 2):
    """
    Creates Data Pipeline to feed the data to the model

    Args:
        X: Image Tensor
        Y: Mask Tensor
        batch: Batch Size
    
    Returns:
        Data Pipeline
    """
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds


if __name__ == "__main__":
    # Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    # Directory for saving for ex. weight file
    create_dir("files")

    # Hyperparameters
    image_h = 256
    image_w = 256
    input_shape = (image_h, image_w, 3)
    batch_size = 4
    lr = 1e-4
    num_epochs = 5

    # Paths 
    dataset_path = r"C:\Users\samru\Study\Study_Abroad\Projects_To_Showcase\Background_Removal\archive\people_segmentation"
    
    # path where model will be saved
    model_path = os.path.join("files", "model.h5")

    # path where csv file will be saved
    csv_path = os.path.join("files", "data.csv")

    # Loading the dataset
    (train_x, train_y), (valid_x, valid_y) = load_dataset(dataset_path, split = 0.2)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)}")
    print(train_y)

    # Dataset Pipeline 
    train_ds = tf_dataset(train_x, train_y, batch = batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch = batch_size)

    # Model 
    model = build_model(input_shape)
    model.compile(
        loss = "binary_crossentropy",
        optimizer = tf.keras.optimizers.Adam(lr)
    )

    # Training 
    callbacks = [
        ModelCheckpoint(model_path, monitor = 'val_loss', verbose = 1, save_best_only = True),
        ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 5, min_lr = 1e-7, verbose = 1),
        CSVLogger(csv_path, append = True),
        EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights = False)
    ]

    model.fit(train_ds,
              validation_data = valid_ds,
              epochs = num_epochs,
              callbacks = callbacks)
