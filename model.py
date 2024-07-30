import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import pickle
from datetime import timedelta, date, datetime


class LstmModel:
    def __init__(self, data : pd.DataFrame, symbol : str, interval):
        self.__interval = interval
        self.__data = data
        self.__symbol = symbol
        self.__scaler = MinMaxScaler(feature_range=(0, 1))
        self.__scalerY = MinMaxScaler(feature_range=(0, 1))
        self.__dataset = data.values
        self.__trainingDataLen = math.ceil(len(self.__dataset) * .8)
        #self.__scaledData = self.__scaler.fit_transform(self.__dataset)
        self.__trainData = self.__dataset[0: self.__trainingDataLen, :]
        self.__testData = self.__dataset[self.__trainingDataLen - 90 :, :]
        self.__xtrain = []
        self.__ytrain = []
        self.__xtest = []
        self.__ytest = []
        self.__model = Sequential()
        self.__saveDate : date
        for i in range(90, len(self.__trainData)):
            self.__xtrain.append(self.__trainData[i - 90 : i])
            self.__ytrain.append(self.__trainData[i, 4])
  

        self.__xtrain = np.array(self.__xtrain)
        self.__ytrain = np.array(self.__ytrain)
        self.__xtrain = np.reshape(self.__xtrain, (self.__xtrain.shape[0], -1))
        self.__xtrain = self.__scaler.fit_transform(self.__xtrain)


        self.__ytrain = np.reshape(self.__ytrain, (self.__ytrain.shape[0], 1))
        self.__xtrain =  np.reshape(self.__xtrain, (self.__xtrain.shape[0], 90, 5))

        self.__ytest =  self.__dataset[self.__trainingDataLen:, 4]

        for i in range(90, len(self.__testData)):
            self.__xtest.append(self.__testData[i - 90: i, : ])

        self.__xtest = np.array(self.__xtest)
        self.__xtest = np.reshape(self.__xtest, (self.__xtest.shape[0], -1))
        self.__xtest = self.__scaler.transform(self.__xtest)
        self.__xtest = np.reshape(self.__xtest, (self.__xtest.shape[0], 90, 5))
        self.__ytrain = self.__scalerY.fit_transform(self.__ytrain)

    def setSymbol(self, symbol):
        self.__symbol = symbol

    def getSymbol(self):
        return self.__symbol
    
    def getDataset(self):
        return self.__dataset
    
    def getxtrain(self):
        return self.__xtrain
    
    def getxtest(self):
        return self.__xtest

    def getytrain(self):
        return self.__ytrain
    
    def getytest(self):
        return self.__ytest
    
    def getdata(self):
        return self.__data
    
    def getModel(self):
        return self.__model

    def buildModel(self):
        self.__model.add(LSTM(64,activation = 'relu', return_sequences = True, input_shape = (self.__xtrain.shape[1], self.__xtrain.shape[2])))
        self.__model.add(LSTM(32,activation = 'relu', return_sequences = False))
        self.__model.add(Dropout(0.2))
        self.__model.add(Dense(self.__ytrain.shape[1]))
        self.__model.compile(optimizer = "adam", loss="mean_squared_error")
        self.__model.fit(self.__xtrain, self.__ytrain, epochs = 20, batch_size = 16, verbose = 1)

    
    def predict(self, data):
        return self.__model.predict(data)
    
    def rmse(self):
        return np.sqrt(np.mean(self.__scalerY.inverse_transform(self.predict(self.__xtest)) - self.__ytest) ** 2)

    def drawGraph(self):
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(16,8))
        plt.title("Close Price History")
        plt.plot(self.__data['Close'])
        plt.xlabel("Date", fontsize = 15)
        plt.ylabel("Close Price USD ($)", fontsize=15)
        plt.show()

    def drawModelGraph(self):
        train = self.__data[:self.__trainingDataLen]
        valid = self.__data[self.__trainingDataLen:]
        valid['Predictions'] = self.__scalerY.inverse_transform(self.predict(self.__xtest))
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(16, 8))
        plt.title("Model")
        plt.xlabel("Date", fontsize = 15)
        plt.ylabel("Close Price USD ($)", fontsize = 15)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc="lower right")
        plt.show()

    def saveModel(self, fileName):
        self.__saveDate = date.today()
        with open(f"{fileName}.pickle", 'wb') as file:
            pickle.dump(self, file)

    def futurePredictions(self):
        lastX = self.__xtest[-1]
        lastX = np.reshape(lastX, (1, 90, 5))
        prediction = self.__scalerY.inverse_transform(self.predict(lastX))
        print(str(self.__saveDate) + " : " + str(prediction))






        

