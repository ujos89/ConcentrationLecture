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
parser.add_argument('--epoch', type=int, default=100, help='number of iterations')
args = parser.parse_args()

#build dataset
#kjk size:2010
def build_dataset(cnt):
    rawdata = pd.read_pickle("./train_set/" + args.file + ".pkl")
    if cnt > len(rawdata):
        return pd.DataFrame()
    elif cnt == 0:
        return rawdata
    else:
        return rawdata[:cnt]
#DNN model

def build_model():
    model = keras.Sequential([
        layers.Dense(len(train_dataset.keys()), activation='relu', input_shape=[len(train_dataset.keys())]),
        #layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    #keras.optimizers.RMSprop(0.1)
    keras.optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    #metrics = ['mae', 'mse','accuracy'])
    return model

#prints dot('.') for every epoch for visual convenience
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

#calculate accuracy
def cal_accuracy(test_labels, test_predictions):
    answer = test_labels.to_numpy()
    tp_set = np.concatenate((np.reshape(test_predictions, (test_predictions.shape[0], 1)), np.reshape(answer, (answer.shape[0], 1))) , axis=1)
    
    #cal standard
    sorted_predictions = np.sort(tp_set)
    standard_idx = np.sum(test_labels)-1
    standard = sorted_predictions[standard_idx,0]
    #print(standard)

    true_cnt = 0
    for tp in tp_set:
        if tp[0] <= standard and tp[1] == 0:
            true_cnt += 1
        elif tp[0] > standard and tp[1] == 1:
            true_cnt += 1

    accuracy = true_cnt / len(test_labels) * 100
    return tp_set, accuracy

#Graph of epoch vs. mse for trainset, val_set 
def plot_history1(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
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
tp_df = pd.DataFrame()
i = 0

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
    tp_set, tmp_accuracy = cal_accuracy(test_labels, test_predictions)
    accuracy.append(tmp_accuracy)
    #print(tp_set[-10:])
    print("accuracy: ",tmp_accuracy,"%")

    #extract tp_set
    column_name = ['predcition('+str(i+1)+'-fold)','label('+str(i+1)+'-fold)']
    tp_tmp = pd.DataFrame(tp_set, columns=column_name)
    tp_df = pd.concat([tp_df, tp_tmp], axis=1)
    i += 1

    #visualize graph
    hist = pd.DataFrame(history.history)
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
tp_df.to_pickle('./analysis/'+args.file+'.pkl')

#print accuracy for each fold
print("accuracy for each fold")
print(accuracy)
print("average accuracy: ",sum(accuracy)/len(accuracy),"%")