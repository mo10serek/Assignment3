# Name this file assignment3.py when you submit
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sklearn


# Keras data generator for the hill vallye dataset
class HillValleyDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataset_filepath, batch_size):
        # dataset_filepath is the path to a .data file containing the dataset
        # batch_size is the batch size for the network
        model = keras.Sequential()
        model.add(keras.layers.Dense(3, imput_shape=(4,), activation="relu"))

        WS = pd.read_excel(dataset_filepath)
        data = np.array(WS)
        x = data[:, :-1]
        self.x = x / x.max(axis=0)
        self.y = data[:, -1]

        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(self.x, self.y, test_size=0.2)

        self.batch_size = batch_size
        self.epochs = 5
        self.model = model

        # Return nothing

    def __len__(self):
        # batches per epoch is the total number of batches used for one epoch

        batches_per_epoch = self.batch_size // self.epochs

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

    model = HillValleyDataGenerator(dataset_filepath, 20)

    model.add(keras.layers.Conv2D(1, 5, input_shape=(100, 100, 1)))



    # model is a trained keras rnn model for predicting compressive strength
    # training_performance is the performance of the model on the training set
    # validation_performance is the performance of the model on the validation set
    return model, training_performance, validation_performance


# A function that creates a keras rnn model to predict whether a sequence has a hill or valley
def hill_valley_rnn_model(dataset_filepath):
    # dataset_filepath is the path to a .data file containing the dataset

    # model is a trained keras rnn model for predicting compressive strength
    # training_performance is the performance of the model on the training set
    # validation_performance is the performance of the model on the validation set
    return model, training_performance, validation_performance