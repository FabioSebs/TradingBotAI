#!/usr/bin/env python
# coding: utf-8

# #### This program uses and artificial recurrent neural network called Long Short Term Memory (LSTM) to predict stock price ####

# In[11]:


#Imports
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# # Stock Data Class 

# In[12]:


# Get the stock quote
class Stock_Data:
    def __init__(self, stock:str, start:str, end:str):
        self.df = web.DataReader(stock, data_source='yahoo', start=start, end=end)
        self.stock = stock
        self.start = start
        self.end = end
        #date example - 2018-01-01
    
    def visualize_date(self):
        #Visualize the closing price history
        plt.figure(figsize=(16,8))
        plt.title('Close Price History of {stock}'.format(stock = self.stock))
        plt.plot(self.df['Close'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.show
        
    def get_training_data_length(self):
        #Create a new dataframe with only the Close Column
        data = self.df.filter(['Close'])
        #Convert the dataframe to a numpy array
        dataset = data.values
        #Get the number of rows to train the models on
        training_data_len = math.ceil(len(dataset)* .8)
        
        return training_data_len
    
    def get_scaled_data(self):
        #Create a new dataframe with only the Close Column
        data = self.df.filter(['Close'])
        #Convert the dataframe to a numpy array
        dataset = data.values
        #Scaling the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        
        return scaled_data
    
    def get_scaler(self):
        #Create a new dataframe with only the Close Column
        data = self.df.filter(['Close'])
        #Convert the dataframe to a numpy array
        dataset = data.values
        #Scaling the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        
        return scaler
    
    def create_training_data(self):
        #Getting Scaled Data and Training Data
        scaled = self.get_scaled_data()
        train_len = self.get_training_data_length()
        
        #Create the training dataset
        train_data = scaled[0:train_len , :]

        #Split the data into x_train and y_train data sets
        x_train = []
        y_train = []

        # X_Train displays the last 60 days for y_train
        for i in range (60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i,0])
            
        return x_train, y_train
    
    def converting_reshaping(self):
        #Getting Training Data
        x_train, y_train = self.create_training_data()
        
        #Convert the x_train and y_train to numpy arrays
        x_train , y_train = np.array(x_train) , np.array(y_train)
        
        #Reshape the data 
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_train.shape
        
        return x_train, y_train
    
    def build_network(self):
        x_train, y_train = self.converting_reshaping()
        
        #Build the LSTM Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        
        #Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        #Train the model
        model.fit(x_train, y_train, batch_size=1, epochs=1)
        
        return model
    
    def testing_data(self):
        #Getting Variables
        scaled_data = self.get_scaled_data()
        training_data_len = self.get_training_data_length()
        
        #Get a Dataset
        data = self.df.filter(['Close'])
        dataset = data.values
        
        #Create the Testing Data Set
        #create a new array containing scaled values form index 1543 to 2003
        test_data = scaled_data[training_data_len - 60: , :]

        #Create the data sets x_test and y_test
        x_test=[]
        y_test= dataset[training_data_len+60:,:]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i,0])
            
        #Convert the data into a numpy array
        x_test = np.array(x_test)
        
        #Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        return x_test, y_test
        
    #xtest, ytest, model
    def predictions(self):
        #Getting Variables
        x_test, y_test = self.testing_data()
        model = self.build_network()
        scaler = self.get_scaler()
        
        #Get the models predicted price values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        
        #Get the root mean squared error (RMSE) (Standard Deviation of the Residuals)
        rmse = np.sqrt( np.mean( predictions - y_test )**2 )
        
        return predictions
    
    def plot_predictions(self):
        #Get Variables
        training_data_len = self.get_training_data_length()
        data = self.df.filter(['Close'])
        predictions = self.predictions()
        
        #Plot the data
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions

        #Visualize the data
        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc="lower right")
        plt.show()


# # Main

# In[14]:

'''
if __name__ == "__main__":
    stock = Stock_Data('TWTR', '2016-06-05', '2021-04-01')
    stock.visualize_date() #works
    stock.plot_predictions()
'''


