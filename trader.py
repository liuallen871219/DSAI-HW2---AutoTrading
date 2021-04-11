import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def normalize(train):
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return train_norm
def buildTrain(train, pastDay=1, futureDay=1):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["open"]))
    return np.array(X_train), np.array(Y_train)
def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]
def buildOneToOneModel(shape):
    model = Sequential()
    model.add(LSTM(50, input_length=shape[1], input_dim=shape[2],return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(50,return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(50,return_sequences=True))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(1)))    # or use model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model

if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    training_data=pd.read_csv(args.training,header=None)
    training_data.columns=["open","high","low","close"]

    testing_data=pd.read_csv(args.testing,header=None)
    testing_data.columns=["open","high","low","close"]
    testing_data=testing_data.values.reshape((testing_data.shape[0],1,4))

    X_train, Y_train = buildTrain(training_data, 1, 1)
    # X_train, Y_train = shuffle(X_train, Y_train)
    Y_train = Y_train[:,np.newaxis]


    model = buildOneToOneModel(X_train.shape)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(X_train, Y_train, epochs=1000, batch_size=64, callbacks=[callback])
    
    min=testing_data[0][0][0]
    max=testing_data[0][0][0]
    a=[]
    p=[]
    rest=1
    hold=0
    sold_price=0
    buy_price=0
    for i in range(len(testing_data)-1):
        t=testing_data[i].reshape((1,1,4))
        pre=model.predict(t)
        p.append(pre[0][0][0])
        if i == 0:
            sold_price=p[i]
            buy_price=p[i]
            a.append(1)
            hold+=1
            rest-=1
            continue
        if p[i]-p[i-1] > 0 and hold > 0    :
            a.append(-1)
            rest+=1
            hold-=1
            sold_price=p[i]
        elif  p[i]-p[i-1] < 0 and rest > 0 :
            a.append(1)
            hold+=1
            rest-=1
            buy_price=p[i]
        else:
            a.append(0)
    output=pd.DataFrame(a)
    output.to_csv("output.csv",header=None,index=False)