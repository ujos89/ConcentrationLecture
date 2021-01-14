import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

parser = argparse.ArgumentParser(description='for running DNN...')
parser.add_argument('--file', type=str, required=True, help='name of file')
parser.add_argument('--plot', type=str, default='0', help='choose plot graph')
parser.add_argument('--size', type=int, default=0, help='choose size of dataset')
parser.add_argument('--epoch', type=int, default=1000, help='number of iterations')
args = parser.parse_args()

#build dataset
#kjk size:2010
def build_dataset(cnt):
    rawdata = pd.read_pickle("../0-data/data_prepared/" + args.file)
    if cnt > len(rawdata):
        return pd.DataFrame()
    elif cnt == 0:
        return rawdata
    else:
        return rawdata[:cnt]
#DNN model

def build_model():
    model = keras.Sequential([
        layers.Dense(len(train_dataset.keys()), activation='sigmoid', input_shape=[len(train_dataset.keys())]),
        layers.Dense(16, activation = 'relu'),
        #layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='relu')
    ])
    #keras.optimizers.RMSprop(0.1)
    keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['mse','accuracy', 'binary_crossentropy'])
    #metrics = ['mae', 'mse','accuracy'])
    return model

#prints dot('.') for every epoch for visual convenience
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

#calculate accuracy and build predcition&label set
def cal_accuracy(test_labels, test_predictions):
    answer = test_labels.to_numpy()
    pl_set = np.concatenate((np.reshape(test_predictions, (test_predictions.shape[0], 1)), np.reshape(answer, (answer.shape[0], 1))) , axis=1)
    '''
    #cal standard
    sorted_predictions = np.sort(pl_set)
    standard_idx = np.sum(test_labels)-1
    standard = sorted_predictions[standard_idx,0]
    #print(standard)
    '''
    standard = 0.5
    true_cnt = 0
    for pl in pl_set:
        if pl[0] <= standard and pl[1] == 0:
            true_cnt += 1
        elif pl[0] > standard and pl[1] == 1:
            true_cnt += 1

    accuracy = true_cnt / len(test_labels) * 100
    return pl_set, accuracy

#Graph of epoch vs. binary_crossentropy for trainset, val_set 
def plot_history1(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.xlabel('Epoch')
    plt.ylabel('binary_crossentropy')
    plt.scatter(hist['epoch'], hist['binary_crossentropy'], label='Train Error')
    plt.scatter(hist['epoch'], hist['val_binary_crossentropy'], label = 'Val Error')
    plt.legend()
    plt.show()

#(optional)Visualized analysis of prediction-TrueValues
def plot_history2(test_labels, test_predictions, test_dataset, model):
    plt.scatter(test_predictions, test_labels)
    plt.xlabel('Predictions')
    plt.ylabel('True Values')
    plt.axis('square')
    #plt.axis('equal')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    _=plt.plot([-100,100], [-100,100])
    plt.show()

#(optional)Visualizes error distribution using histogram
def plot_history3(test_predictions, test_labels, hist):
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _=plt.ylabel("Count")
    plt.show()

#historgram for test prediction

##main

#split test, train set(X: train_dataset)
dataset = build_dataset(args.size)
X = dataset.sample(frac=0.8, random_state = 0)
test_dataset = dataset.drop(X.index)

#split label(y: train_label)
y = X.pop('label')
test_labels = test_dataset.pop('label')

# Stratified k-fold
skf = StratifiedKFold(n_splits=5, random_state=42)

#deep learning for each k-folded set
accuracy = []
pl_df = pd.DataFrame()

for train_idx, val_idx in skf.split(X, y): 
    train_dataset, val_dataset = X.iloc[train_idx], X.iloc[val_idx]
    train_labels, val_labels = y.iloc[train_idx], y.iloc[val_idx]

    #run dnn
    model = build_model()
    #print(model.summary())
    history = model.fit(train_dataset, train_labels, epochs=args.epoch, validation_data=(val_dataset, val_labels), verbose=0, callbacks=[PrintDot()])
    #save model
    #model.save('model_pose.model')

    #calculate prediction
    test_predictions = model.predict(test_dataset).flatten()
    pl_set, tmp_accuracy = cal_accuracy(test_labels, test_predictions)
    accuracy.append(tmp_accuracy)
    print(pl_set[-100:])
    print("accuracy: ",tmp_accuracy,"%")

    #extract pl_set
    pl_tmp = pd.DataFrame(pl_set, columns=['prediction','label'])
    pl_df = pd.concat([pl_df, pl_tmp], axis=0)

    #visualize graph
    hist = pd.DataFrame(history.history)
    print(hist)
    hist['epoch']= history.epoch
    plot = list(map(int, args.plot.split(',')))
    for idx in plot:
        if idx == 1:
            plot_history1(history)
        elif idx == 2:
            plot_history2(test_labels, test_predictions, test_dataset, model)
        elif idx == 3:
            plot_history3(test_predictions, test_labels, hist)

#save predictions label to pickle
pl_df.to_pickle('../0-data/data_prediction/'+args.file+'_pl.pkl')

#print accuracy for each fold
print("accuracy for each fold")
print(accuracy)
print("average accuracy: ",sum(accuracy)/len(accuracy),"%")