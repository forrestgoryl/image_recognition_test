import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import dataset
fashion_mnist = keras.datasets.fashion_mnist

# split dataset into training and testing sets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# preprocess training and testing sets (datapoints should be between 0 and 1)
train_images = train_images / 255.0
test_images = test_images / 255.0

def run_nn_model(epochs, hidden_layer_neurons):
    if len(hidden_layer_neurons) == 1:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(hidden_layer_neurons[0], activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])
    elif len(hidden_layer_neurons) == 2:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(hidden_layer_neurons[0], activation="relu"),
            keras.layers.Dense(hidden_layer_neurons[1], activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])
    elif len(hidden_layer_neurons) == 3:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(hidden_layer_neurons[0], activation="relu"),
            keras.layers.Dense(hidden_layer_neurons[1], activation="relu"),
            keras.layers.Dense(hidden_layer_neurons[2], activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])
    elif len(hidden_layer_neurons) == 4:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(hidden_layer_neurons[0], activation="relu"),
            keras.layers.Dense(hidden_layer_neurons[1], activation="relu"),
            keras.layers.Dense(hidden_layer_neurons[2], activation="relu"),
            keras.layers.Dense(hidden_layer_neurons[3], activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])
    elif len(hidden_layer_neurons) == 5:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(hidden_layer_neurons[0], activation="relu"),
            keras.layers.Dense(hidden_layer_neurons[1], activation="relu"),
            keras.layers.Dense(hidden_layer_neurons[2], activation="relu"),
            keras.layers.Dense(hidden_layer_neurons[3], activation="relu"),
            keras.layers.Dense(hidden_layer_neurons[4], activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])

    model.compile(optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    model.fit(train_images, train_labels, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

    return test_acc

hidden_layer_neurons = []
for arg in sys.argv[1:]:
    hidden_layer_neurons.append(int(arg))

results = open("results.txt", "a")

results.write(f"\nresults with {hidden_layer_neurons} neurons in {len(hidden_layer_neurons)} hidden layer(s)\n\n")
for i in range(1, 3):
    test_acc = run_nn_model(i * 5, hidden_layer_neurons)
    results.write(f"test acc @ {i*5} epochs: {test_acc}\n")

results.close()