import pandas as pd
import numpy as np
import yfinance as yf
from model import LstmModel
import datetime
from dateutil.relativedelta import relativedelta
import pickle


while True:

    print("*****MENU*****")
    print("Type 1 to save a model")
    print("Type 2 to pick a model")
    print("Type 3 to exit")


    a = int(input("Enter a number : "))
    if a == 1:
        symbol = input("enter a symbol(EX:AAPL) : ")
        years = int(input("enter time period in years(EX:5) : "))
        fileName = input("Enter a name for model : ")
        end = datetime.date.today()
        start = end - relativedelta(years=years)
        data = yf.download(symbol, start=str(start), end=str(end))
        data = data.filter(['Close'])
        model = LstmModel(data, symbol)
        model.buildModel()
        model.saveModel(fileName)

    if a == 2:
        fileName = input("Enter the model name : ")
        try:
            with open(f"{fileName}.pickle", "rb") as file:
                loadedModel = pickle.load(file)
        except:
            print("Error.")
            pass

        print(f"***SELECTED MODEL : {fileName}***")
        print("Type 1 to draw price history graph")
        print("Type 2 to draw graph with test prediction")
        print("Type 3 to see future predictions")
        print("Type 4 to exit to menu")
        while True:
            opt = int(input("Enter a number : "))
            if opt == 1:
                loadedModel.drawGraph()
            if opt == 2:
                loadedModel.drawModelGraph()

            if opt == 3:
                print(loadedModel.futurePredictions(10))
            if opt == 4:
                break
    if a == 3:
        exit()
