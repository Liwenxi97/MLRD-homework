import nltk
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

"""data processing"""

"""Model Traning"""
# 损失函数公式
def compute_cost(W,X,Y):
    N = X.shape[0]
    distance = 1 - Y * (np.dot(X,W))
    distance[distance<0] = 0
    hinge_loss = regularization * (np.sum(distance) / N)

    # calculate the cost
    cost = 0.5 * np.dot(W, W) + hinge_loss

    return cost

# 计算损失函数斜率
def calculate_cost_gradient(W, X_batch, Y_batch):
    if type(Y_batch) = np.float64:
        X_batch = np.array([X_batch])
        Y_batch = np.array([Y_batch])
    distance = 1 - (Y_batch * np.dot(W, X_batch))
    dw = np.zero(len(W))

    for ind,dis in enumerate(distance):
        if max(0,dis) == 0:
            di = W
        else:
            di = W - (regularization * Y_batch[ind] * X_batch[ind])
        dw += di


    dw = dw / len(Y_batch)
    return dw


# 模型训练
def sgd(features, outputs):
    #
    max_epoch = 5000
    weight = np.zero(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    threshold  = 0.01

    for epoch in range(1,max_epoch):
        X, Y = shuffle(features,outputs)
        for ind, d in enumerate(features):
            ascent = calculate_cost_gradient(weight, d, Y[ind])
            weight = weight - (learning_rate * ascent)

        if epoch == 2 ** nth or epoch == max_epoch - 1:
            cost = compute(weight, features, outputs)
            if abs(prev_cost-cost) < prev_cost * threshold:
                return weight
            prev_cost = cost
            nth += 1

    return weight

def init():
    print("reading dataset...")
    # read data in pandas (pd) data frame
    data = pd.read_csv('./data/data.csv')

    # drop last column (extra column added by pd)
    # and unnecessary first column (id)
    data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

    print("applying feature engineering...")
    # convert categorical labels to numbers
    diag_map = {'M': 1.0, 'B': -1.0}
    data['diagnosis'] = data['diagnosis'].map(diag_map)

    # put features & outputs in different data frames
    Y = data.loc[:, 'diagnosis']
    X = data.iloc[:, 1:]


    # normalize data for better convergence and to prevent overflow
    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)

    # insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)

    # split data into train and test set
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)

    # train the model
    print("training started...")
    W = sgd(X_train.to_numpy(), y_train.to_numpy())
    print("training finished.")
    print("weights are: {}".format(W))

    # testing the model
    print("testing the model...")
    y_train_predicted = np.array([])
    for i in range(X_train.shape[0]):
        yp = np.sign(np.dot(X_train.to_numpy()[i], W))
        y_train_predicted = np.append(y_train_predicted, yp)

    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(X_test.to_numpy()[i], W))
        y_test_predicted = np.append(y_test_predicted, yp)

    print("accuracy on test dataset: {}".format(accuracy_score(y_test, y_test_predicted)))
    print("recall on test dataset: {}".format(recall_score(y_test, y_test_predicted)))
    print("precision on test dataset: {}".format(recall_score(y_test, y_test_predicted)))   

regularization_strength = 10000
learning_rate = 0.000001