# Name this file assignment3.py when you submit
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import os

cwd = os.getcwd()


# Keras data generator for the hill vallye dataset
class HillValleyDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_filepath, batch_size):
        # dataset_filepath is the path to a .data file containing the dataset
        # batch_size is the batch size for the network
        # print(dataset_filepath)

        self.data = pd.read_csv(dataset_filepath, sep=",")

        # print(self.data)

        self.x = self.data.values[:, 0:-1]
        self.y = self.data['class'].values
        self.x = np.reshape(self.x, (606, 100))
        print(self.x.shape)
        # print(self.y.shape)

        # print(self.x)
        # print(self.y)

        self.number_of_images = len(self.x)
        self.batch_size = batch_size
        self.epochs = 5

        # Return nothing

    def __len__(self):
        # batches per epoch is the total number of batches used for one epoch

        batches_per_epoch = self.number_of_images // self.batch_size

        return batches_per_epoch

    def __getitem__(self, index):
        # index is the index of the batch to be retrieved

        batch_of_x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        batch_of_y = self.y[index * self.batch_size:(index + 1) * self.batch_size]

        # x is one batch of data
        # y is the labels associated with the batch
        return batch_of_x, batch_of_y


# A function that creates a keras cnn model to predict whether a sequence has a hill or valley
def hill_valley_cnn_model(dataset_filepath):
    # dataset_filepath is the path to a .data file containing the dataset

    testHillValleyGenerator = HillValleyDataGenerator(dataset_filepath + '/Hill_Valley_with_noise_Testing.data', 20)
    trainingHillValleyGenerator = HillValleyDataGenerator(dataset_filepath + '/Hill_Valley_with_noise_Training.data',
                                                          20)

    model = keras.Sequential()

    model.add(keras.layers.Conv1D(filters=16, kernel_size=5, activation='sigmoid', input_shape=(606, 100,)))
    # model.add(keras.layers.MaxPooling2D(pool_size=2))
    # model.add(keras.layers.Dense(256, activation="relu"))
    # model.add(keras.layers.Dense(102, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    training_performance = model.fit(trainingHillValleyGenerator.x,
                                     trainingHillValleyGenerator.y,
                                     epochs=20,
                                     verbose=2)

    validation_performance = model.evaluate(testHillValleyGenerator.x,
                                            testHillValleyGenerator.y)

    # model is a trained keras rnn model for predicting compressive strength
    # training_performance is the performance of the model on the training set
    # validation_performance is the performance of the model on the validation set
    return model, training_performance, validation_performance


hill_valley_cnn_model(cwd)


# A function that creates a keras rnn model to predict whether a sequence has a hill or valley
def hill_valley_rnn_model(dataset_filepath):
    # dataset_filepath is the path to a .data file containing the dataset

    # model is a trained keras rnn model for predicting compressive strength
    # training_performance is the performance of the model on the training set
    # validation_performance is the performance of the model on the validation set
    return model, training_performance, validation_performance