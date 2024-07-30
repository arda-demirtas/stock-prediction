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
        self.__dataset = data.values
        self.__trainingDataLen = math.ceil(len(self.__dataset) * .8)
        self.__scaledData = self.__scaler.fit_transform(self.__dataset)
        self.__trainData = self.__scaledData[0: self.__trainingDataLen, :]
        self.__testData = self.__scaledData[self.__trainingDataLen - 60 :, :]
        self.__xtrain = []
        self.__ytrain = []
        self.__xtest = []
        self.__ytest = []
        self.__model = Sequential()
        self.__saveDate : date

        for i in range(60, len(self.__trainData)):
            self.__xtrain.append(self.__trainData[i - 60 : i, 0])
            self.__ytrain.append(self.__trainData[i, 0])

        self.__xtrain = np.array(self.__xtrain)
        self.__ytrain = np.array(self.__ytrain)

        self.__xtrain =  np.reshape(self.__xtrain, (self.__xtrain.shape[0], self.__xtrain.shape[1], 1))

        self.__ytest =  self.__dataset[self.__trainingDataLen:, :]

        for i in range(60, len(self.__testData)):
            self.__xtest.append(self.__testData[i - 60: i, : ])

        self.__xtest = np.array(self.__xtest)
        self.__xtest = np.reshape(self.__xtest, (self.__xtest.shape[0], self.__xtest.shape[1], 1))

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
        self.__model.add(LSTM(50, return_sequences = True, input_shape = (self.__xtrain.shape[1], 1)))
        self.__model.add(Dropout(0.2))
        self.__model.add(LSTM(50, return_sequences = False))
        self.__model.add(Dropout(0.2))
        self.__model.add(Dense(25))
        self.__model.add(Dense(1))
        self.__model.compile(optimizer = "adam", loss="mean_squared_error")
        self.__model.fit(self.__xtrain, self.__ytrain, epochs = 100, batch_size = 32, verbose = 1)

    def inverse(self, data):
        return self.__scaler.inverse_transform(data)
    
    def predict(self, data):
        return self.__model.predict(data)
    
    def rmse(self):
        return np.sqrt(np.mean(self.inverse(self.predict(self.__xtest)) - self.__ytest) ** 2)

    def drawGraph(self):
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(16,8))
        plt.title("Close Price History")
        plt.plot(self.__data)
        plt.xlabel("Date", fontsize = 15)
        plt.ylabel("Close Price USD ($)", fontsize=15)
        plt.show()

    def drawModelGraph(self):
        train = self.__data[:self.__trainingDataLen]
        valid = self.__data[self.__trainingDataLen:]
        valid['Predictions'] = self.inverse(self.predict(self.__xtest))
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
        self.__saveDate = datetime.today()
        with open(f"{fileName}.pickle", 'wb') as file:
            pickle.dump(self, file)

    def futurePredictions(self, numberOfNextPredictions):
        lastTest = self.__xtest[-1]
        lastTest = np.reshape(lastTest, (1, 60, 1))
        nextPredictionList = []
        dates = []
        saveDate = self.__saveDate
        
        match self.__interval[1]:
            case 'h':
                td = timedelta(hours=int(self.__interval[0]))
            case 'm':
                td = timedelta(minutes=int(self.__interval[0]))
            case 'd':
                td = timedelta(days=int(self.__interval[0]))
            
            
        for i in range(numberOfNextPredictions):
            nextPrediction = self.predict(lastTest)
            nextPredictionList.append(nextPrediction[0, 0])
            lastTest = np.roll(lastTest, -1, axis=1)
            lastTest[0, -1] = nextPrediction
                
            dates.append(str(saveDate + i * td))
        dates = np.array(dates)
        dates = np.reshape(dates, (numberOfNextPredictions, 1))
        nextPredictionList = self.inverse(np.array(nextPredictionList).reshape(-1, 1))
        df_vertical = pd.DataFrame(np.hstack((dates, nextPredictionList)), columns=['Date', 'Prediction'])
        df_vertical = df_vertical.set_index('Date')
        df_vertical.index = df_vertical.index.astype("datetime64[ns]")
        df_vertical['Prediction'] = df_vertical['Prediction'].astype('Float64')
        print(df_vertical)
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(16, 8))
        plt.title("Model")
        plt.xlabel("Date", fontsize = 15)
        plt.ylabel("Close Price USD ($)", fontsize = 15)
        plt.plot(df_vertical['Prediction'])
        plt.show()







        

