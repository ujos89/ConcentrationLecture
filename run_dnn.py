import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='for running DNN...')
parser.add_argument('--file', type=str, required=True, help='name of file')
parser.add_argument('--plot', type=str, default='2', help='choose plot graph')
args = parser.parse_args()

dataset = pd.read_pickle("./train_set/" + args.file + ".pkl")

train_dataset = dataset.sample(frac=0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

#remove label from dataset
train_labels = train_dataset.pop('label')
test_labels = test_dataset.pop('label')

def build_model():
    model = keras.Sequential([
        layers.Dense(18, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),

        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return model

model = build_model()

#(optional) shows model info
model.summary()

#(optional) checks if model is working properly
example_batch = train_dataset[:10]
example_result = model.predict(example_batch)
example_result

#(optional) prints dot('.') for every epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

#number of epochs
EPOCHS = 1000

#(model.fit==> what is this??)
history = model.fit(
    train_dataset, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()]
)

#Saves model
model.save('model_pose01.model')

#(optional) visualizes training process
hist = pd.DataFrame(history.history)
hist['epoch']= history.epoch
test_predictions = model.predict(test_dataset).flatten()

#(optional) Graph of training process
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.scatter(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.scatter(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
    plt.ylim([0,.5])
    plt.legend()
    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.scatter(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.scatter(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
    plt.legend()
    plt.show()

#(optional)Visualizing evaluation of model using test data
def plot_history2(test_labels, test_predictions, test_dataset, model):
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _=plt.plot([-100,100], [-100,100])
    plt.show()

#(optional)Visualizes error distribution using histogram
def plot_history3(test_predictions, test_labels, hist):
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _=plt.ylabel("Count")
    plt.show()

#plot graph
plot = list(map(int, args.plot.split(',')))
for idx in plot:
    if idx == 1:
        plot_history(history)
    elif idx == 2:
        plot_history2(test_labels, test_predictions, test_dataset, model)
    elif idx == 3:
        plot_history3(test_predictions, test_labels, hist)

#calculate accuracy
answer = test_labels.to_numpy()
tp_set = np.concatenate((np.reshape(test_predictions, (test_predictions.shape[0], 1)), np.reshape(answer, (answer.shape[0], 1))) , axis=1)
true_cnt = 0
for tp in tp_set:
    if tp[0] <= 0.5 and tp[1] == 0:
        true_cnt += 1
    elif tp[0] > 0.5 and tp[1] == 1:
        true_cnt += 1

accuracy = true_cnt / len(test_labels) * 100
print(tp_set[-10:])
print("accuracy: ",accuracy,"&")