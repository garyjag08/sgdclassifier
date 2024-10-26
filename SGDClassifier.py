import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
import yfinance as yf

class SetStockData:
  def __init__(self, stock_name):
    self.stock_name = stock_name
  
  def set_stock_data(self, start_date, end_date):
    stock_data = yf.download(self.stock_name, start=start_date, end=end_date)
    return stock_data

  def setDirection(self, df):
    df["Direction"] = np.where(df["Close"] > df["Close"].shift(1), "up", "down")
    return df

  def setPercentChange(self, df):
    df["Change_Open_Close"] = df["Open"] - df["Close"].shift(1)
    return df

  def  setdataFrame(self, start_date, end_date):
    df = self.set_stock_data(start_date, end_date)
    df = self.setDirection(df)
    df = self.setPercentChange(df)
    df = df.iloc[1:] # exclude the first row due to NaN values
    return df

class SGDClassification:
  def __init__(self):
    self.sgd = SGDClassifier()

  def train_model(self, df, x_attributes, y_attribute):
    x = df[x_attributes]
    y = df[y_attribute]
    trained_model = self.sgd.fit(x, y)
    return trained_model

  def predict(self, trained_model, x_attributes):
    prediction = trained_model.predict(x_attributes)
    return prediction

# set dataframe, SPY from January 29, 2024 through October 25, 2024
stock_name = "SPY"
start_date = "2024-01-29"
end_date = "2024-10-25"
stock_data_spy = SetStockData(stock_name)
df = stock_data_spy.setdataFrame(start_date, end_date)

df.head()

# set X attributes
x_attributes = ["Change_Open_Close"]

# set Y attribute
y_attribute = "Direction"

x_new = [[1.05]]

trained_model = SGDClassification().train_model(df, x_attributes, y_attribute)
prediction = SGDClassification().predict(trained_model, x_new)
print(prediction[0])

