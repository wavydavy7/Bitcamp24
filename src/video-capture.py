# import the opencv library 
import cv2
import numpy as np
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
from keras.models import load_model

# define a video capture object 
vid = cv2.VideoCapture(0) 

emotions = ['HAPPY', 'SAD']

dir = f"./model/"
model = load_model(f'{dir}model.h5')
model.load_weights(f'{dir}weights.h5')

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Get the dimensions of the original image
    frame_matrix = np.array(frame)
    # print(frame_matrix.shape)
    # print(frame_matrix)
    height, width, _ = frame_matrix.shape

    # Calculate the size of the square (assume you want to crop a square with size min(height, width))
    size = int(min(height, width)/1.5)

    # Calculate the starting position for cropping
    start_x = (width - size) // 2
    start_y = (height - size) // 2

    # Crop the middle square
    cropped_image = frame_matrix[start_y:start_y+size, start_x:start_x+size]

    gray_image = cv2.cvtColor(frame_matrix, cv2.COLOR_BGR2GRAY)

    # Resize the image to 48x48
    resized_gray_image = cv2.resize(gray_image, (48, 48))

    input_image = np.reshape(resized_gray_image, (1, 48, 48))

    # Make predictions
    predictions = model.predict(input_image)

    # Assuming it's a classification task with multiple classes, you may want to get the class with the highest probability
    predicted_class = np.argmax(predictions)

    print(predicted_class)

    # Print the predicted class
    print("Predicted class:", emotions[predicted_class])

    cv2.imwrite('saved_image.jpg', resized_gray_image)

    text_frame = cropped_image
    text = emotions[predicted_class]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color in BGR format
    thickness = 2

    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the position to put the text (bottom-left corner)
    text_x = (cropped_image.shape[1] - text_size[0]) // 2
    text_y = cropped_image.shape[0] - 10

    # Put the text on the image
    cv2.putText(text_frame, text, (text_x, text_y), font, font_scale, font_color, thickness)

    # Display the resulting frame 
    cv2.imshow('frame', cropped_image) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 