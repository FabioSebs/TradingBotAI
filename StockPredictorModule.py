#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[4]:


import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import dotenv
dotenv.load_dotenv()
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from numpy import array


# # Stock Prediction Module

# In[16]:


class Stock_Predict():
    def __init__(self, stock:str):
        self.key = os.getenv("KEY")
        self.stock = stock
        self.df = pdr.get_data_tiingo(self.stock, api_key = self.key)
        
    def get_df(self):
        self.df.to_csv(f'{self.stock}.csv')
        df = pd.read_csv(f'{self.stock}.csv')
        df1 = df.reset_index()['close']
        return(df1)
        
        
    def plot_stock(self):
        df1 = self.get_df()
        plt.plot(df1)
        
    def get_scaled_data(self):
        df1 = self.get_df()
        scaler = MinMaxScaler(feature_range=(0,1))
        df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
        return(df1, scaler)
    
    def get_traintest_size(self):
        df1, scaler = self.get_scaled_data()
        training_size = int(len(df1)*0.65)
        training_size = len(df1)-training_size
        train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1),:1]
        return(train_data, test_data)
    
    def create_dataset(self, dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step),0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    def get_traintest_data(self):
        train_data, test_data = self.get_traintest_size()
        time_step = 100
        X_train, Y_train = self.create_dataset(train_data, time_step)
        X_test, Y_test = self.create_dataset(test_data, time_step)
        return(X_train, Y_train, X_test, Y_test)
    
    def reshaped_data(self):
        X_train, Y_train, X_test, Y_test = self.get_traintest_data()
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        return(X_train, X_test, Y_train, Y_test)
    
    def build_model(self):
        model=Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer='adam')
        return(model)
    
    def fit_model(self):
        X_train, X_test, Y_train, Y_test = self.reshaped_data()
        
        model = self.build_model()
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=64, verbose=1)
        return(model)
        
    def get_predictions(self):
        X_train, X_test, Y_train, Y_test = self.reshaped_data()
        model = self.fit_model()
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        return(train_predict, test_predict)
    
    def reverse_scaling(self):
        df1, scaler = self.get_scaled_data()
        scaler = MinMaxScaler(feature_range=(0,1))
        df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
        train_predict, test_predict = self.get_predictions()
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)
        return(train_predict, test_predict)
        
    def calculate_rmse(self):
        X_train, X_test, Y_train, Y_test = self.reshaped_data()
        train_predict, test_predict = self.reverse_scaling()
        print(math.sqrt(mean_squared_error(Y_train, train_predict)))
        print(math.sqrt(mean_squared_error(Y_test, test_predict)))
        
    def plot_predictions(self):
        X_train, Y_train, X_test, Y_test = self.get_traintest_data()
        train_predict, test_predict = self.reverse_scaling()
        df1, scaler = self.get_scaled_data()
        # shift train predictions for plotting
        look_back=100
        trainPredictPlot = np.empty_like(df1)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(df1)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
        # plot baseline and predictions
        plt.plot(scaler.inverse_transform(df1))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.show()
        
    def predict_future(self):
        train_data, test_data = self.get_traintest_size()
        x_input = test_data[-100:].reshape(1,-1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        model = self.fit_model()
        lst_output=[]
        n_steps=100
        i=0
        while(i<30):
            if(len(temp_input)>100):
                #print(temp_input)
                x_input = np.array(temp_input[1:])
                #print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                #print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                #print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1
        
        day_new=np.arange(1,101)
        day_pred=np.arange(101,131)
        
        return (day_new, day_pred, lst_output)
    
    def plot_future_predictions(self):
        df1, scaler = self.get_scaled_data()
        day_new, day_pred, lst_output = self.predict_future()
        plt.plot(day_new,scaler.inverse_transform(df1[-100:]))
        plt.plot(day_pred,scaler.inverse_transform(lst_output))
        
        df3=df1.tolist()
        df3.extend(lst_output)
        plt.plot(df3[1200:])
        
        df3=scaler.inverse_transform(df3).tolist()
        plt.plot(df3)
    
    def get_gain_loss(self):
        df1 = self.get_df()
        end = df1.tail()
        end = end.to_string(index=False)
        numbers = end.split()
        evaluate = float(numbers[0])- float(numbers[-1])
        if evaluate>0:
            gain = 1
            loss = 0
        else:
            gain = 0
            loss = 1
            
        return(gain,loss)

# In[17]:


if __name__ == "__main__":
    predict_stock = Stock_Predict("FB")
    predict_stock.plot_stock()
    #predict_stock.plot_predictions()
    predict_stock.plot_future_predictions()


# In[ ]:





# In[ ]:




