# Name this file assignment3.py when you submit
import tensorflow as tf

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
    x = x / x.max(axis=0)
    y = data[:, -1]

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="mean_squared_error")

    model.fit(x=x_train, y=y_train, batch_size=batch_size, validation_split=0.2, epochs=batch_size, verbose=2)

    # Return nothing    

  def __len__(self):
    #batches per epoch is the total number of batches used for one epoch
    return batches_per_epoch


  def __getitem__(self, index):
    # index is the index of the batch to be retrieved
      
    # x is one batch of data
    # y is the labels associated with the batch
    return x, y


# A function that creates a keras cnn model to predict whether a sequence has a hill or valley
def hill_valley_cnn_model(dataset_filepath):
  # dataset_filepath is the path to a .data file containing the dataset

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