from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import csv
import os
import time
import pandas as pd
import tensorflow as tf
from keras import layers, models, callbacks
from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import cv2


emotions = ['happy', 'sad']

def main():
    print(f"Processing happy images...")

    happy_dir = f"./dataset/happy/"

    # Get the list of image file names in the directory
    image_files = os.listdir(happy_dir)

    # List to store the images
    images = []

    num = 0

    # Read each image and append it to the list
    for image_file in image_files:
        # Read the image
        image_path = os.path.join(happy_dir, image_file)
        image = cv2.imread(image_path)
        
        # Check if the image was read successfully
        if image is not None:
            images.append(image)

        if (num % 100 == 0):
            print(f"On image {num}")
        num += 1

    # Convert the list of images to a NumPy array
    images_array = np.array(images)

    # Update console
    print(f"Saving numpy array to file!")
    np.save(f'./dataset/happy.npy', images_array)

    print(f"Processing sad images...")
    sad_dir = f"./dataset/sad/"

    # Get the list of image file names in the directory
    image_files = os.listdir(sad_dir)

    # List to store the images
    images = []

    num = 0

    # Read each image and append it to the list
    for image_file in image_files:
        # Read the image
        image_path = os.path.join(sad_dir, image_file)
        image = cv2.imread(image_path)
        
        # Check if the image was read successfully
        if image is not None:
            images.append(image)

        if (num % 100 == 0):
            print(f"On image {num}")
        num += 1

    # Convert the list of images to a NumPy array
    images_array = np.array(images)

    # Update console
    print(f"Saving numpy array to file!")
    np.save(f'./dataset/sad.npy', images_array)

    # print(images_array.shape)

    print(f"Processing neutral images...")
    neutral_dir = f"./dataset/neutral/"

    # Get the list of image file names in the directory
    image_files = os.listdir(neutral_dir)

    # List to store the images
    images = []

    num = 0

    # Read each image and append it to the list
    for image_file in image_files:
        # Read the image
        image_path = os.path.join(neutral_dir, image_file)
        image = cv2.imread(image_path)
        
        # Check if the image was read successfully
        if image is not None:
            images.append(image)

        if (num % 100 == 0):
            print(f"On image {num}")
        num += 1

    # Convert the list of images to a NumPy array
    images_array = np.array(images)

    # Update console
    print(f"Saving numpy array to file!")
    np.save(f'./dataset/neutral.npy', images_array)

    print(images_array.shape)

if __name__ == '__main__':
    main()

