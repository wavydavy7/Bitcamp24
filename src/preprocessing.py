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
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pennylane as qml

emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

def main():
    df = pd.read_csv("./dataset/data.csv").drop(columns=["Usage"])

    df = df[df["emotion"].isin([3, 4])]
    
    labels = df["emotion"].values
    image_strings = df["pixels"].values
    image_vector = np.array([int(x) for s in image_strings for x in s.split()])
    images = image_vector.reshape(labels.shape[0], 48, 48)

    print(images.shape)
    print(labels.shape)

    # print(labels[0])
    # plt.imshow(images[0], cmap='gray')  # You can change the colormap as needed
    # plt.axis('off')  # Turn off axis
    # plt.show()

    train_images, test_images, train_labels, test_labels \
        = train_test_split(images, labels, test_size=0.1, random_state=42)

    n_layers = 6
    weight_shapes = {"weights": (n_layers, n_qubits)}
    qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

    model = models.Sequential()
    clayer_1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1))
    clayer_2 = layers.MaxPooling2D((2, 2))
    clayer_3 = layers.Conv2D(64, (3, 3), activation='relu')
    clayer_4 = layers.MaxPooling2D((2, 2))
    clayer_5 = layers.Conv2D(128, (3, 3), activation='relu')
    clayer_6 = layers.Flatten()
    clayer_7 = layers.Dense(128, activation='relu')
    clayer_9 = layers.Dense(7, activation='softmax')

    # clayer_1 = tf.keras.layers.Dense(2)
    # clayer_2 = tf.keras.layers.Dense(2, activation="softmax")
    model = tf.keras.models.Sequential([clayer_1, 
                                        clayer_2,
                                        clayer_3,
                                        clayer_4,
                                        clayer_5,
                                        clayer_6,
                                        clayer_7,
                                        qlayer, 
                                        clayer_9])

    # Store the summary
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    SUMMARY = "\n".join(summary_list)
    print(SUMMARY)

    callback = callbacks.EarlyStopping(
        monitor="val_accuracy", baseline=0.8, verbose=1, 
        patience=30, restore_best_weights=True)

    model.compile(optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=["accuracy"])

    # Number of epochs
    NUM_EPOCHS = 200

    # Get training start time
    start_time = time.time()

    # Train the model
    model.fit(train_images, train_labels, epochs=NUM_EPOCHS, 
            validation_data=(test_images, test_labels), 
            callbacks=[callback])
    
    # Get training end time
    end_time = time.time()

    # Test the model and get metrics
    test_loss, test_acc = model.evaluate(test_images, test_labels, 
                                        verbose=2)

    # Update console
    print(f"Average test accuracy: {test_acc}")
    print(f"Average test loss: {test_loss}")
    print(f"Average Fit time: {end_time - start_time}")

    # Create a dataframe to store results to be saved into file
    results = pd.DataFrame()
    results["Summary"] = [SUMMARY]
    results["Number of Epochs"] = [NUM_EPOCHS]
    results["Fit Time"] = [end_time - start_time]
    results["Test Accuracy"] = [test_acc]
    results["Test Loss"] = [test_loss]
    
    # Transpose the dataframe to get metrics as rows
    results = results.T

    # Update console
    print("Results: ")
    print(results)


if __name__ == '__main__':
    main()

