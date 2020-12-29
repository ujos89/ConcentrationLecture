import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset = pd.read_pickle("data_pickle/test.pkl")

train_dataset = dataset.sample(frac=0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

#remove label from dataset
train_labels = train_dataset.pop('label')
test_labels_test_dataset.pop('label')

def build_model():
    model = keras.Sequential([
        layers.Dense(18, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, actiavtion='sigmoid'),
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
EPOCHS = 10

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
hist.tail()

#(optional) Graph of training process
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.scatter(hist['epoch'], hist['mae'], label='Train Error')
    plt.scatter(hist['epoch'], hist['val_mae'], label = 'Val Error')
    plt.ylim([0,.5])
    plt.legend()
    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.scatter(hist['epoch'], hist['mse'], label='Train Error')
    plt.scatter(hist['epoch'], hist['val_mse'], label = 'Val Error')
    plt.legend()
    plt.show()
plot_history(history)

#Visualizing evaluation of model using test data
test_predictions = model.predict(test_dataset).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_=plt.plot([-100,100], [-100,100])
plt.show()

#Visualizes error distribution using histogram
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_=plt.ylabel("Count")
plt.show()

#print error
print(test_labels)
print(test_predictions)