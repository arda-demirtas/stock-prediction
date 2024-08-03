from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter('ignore')
import yfinance as yf
from model import LstmModel
import datetime
from dateutil.relativedelta import relativedelta
import pickle
import tensorflow as tf

def loadModel(fileName):
    with open(f"{fileName}.pickle", "rb") as file:
        loadedModel = pickle.load(file)

    return loadedModel


while True:

    print("*****MENU*****")
    print("Type 1 to save a model")
    print("Type 2 to pick a model")
    print("Type 3 to exit")


    a = int(input("Enter a number : "))
    if a == 1:
        symbol = input("enter a symbol(EX:AAPL) : ")
        months = int(input("enter time period in months(EX:48) : "))
        fileName = input("Enter a name for model : ")
        end = datetime.date.today()
        start = end - relativedelta(months=months)
        data = yf.download(symbol, start=str(start), end=str(end), interval="1d")
        data = data[['Open', 'High', 'Low', 'Volume', 'Close']]
        model = LstmModel(data, symbol, "1d")
        model.buildModel()
        model.saveModel(fileName)


    if a == 2:
        fileName = input("Enter the model name : ")
        loadedModel = loadModel(fileName)

        if loadedModel:
            pass
        else:
            print("Error.")
            exit()

        print(f"***SELECTED MODEL : {fileName}***")
        print("Type 1 to draw price history graph")
        print("Type 2 to draw graph with test prediction")
        print("Type 3 to see future predictions")
        print("Type 4 to see RMSE")
        print("Type 5 to exit to menu")
        while True:
            opt = int(input("Enter a number : "))
            if opt == 1:
                loadedModel.drawGraph()
            if opt == 2:
                loadedModel.drawModelGraph()
            if opt == 3:
                loadedModel.futurePredictions()
            if opt == 4:
                print("RMSE : " + str(loadedModel.rmse()))
            if opt == 5:
                break
            
    if a == 3:
        exit()
