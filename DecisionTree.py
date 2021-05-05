#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import csv as cv
import StockPredictorModule as spm
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# df = pd.read_csv("TWTR.csv")
# df.tail()


# In[2]:


# stocklist = ["AAPL", "NFLX", "MSFT", "SBUX"]
# add_csv_data(stocklist)


# In[3]:


class CSV_Maker():
        
        def __init__(self):
            pass
        
        def make_csv(self):
            file = open("stock.csv", "a", newline="")
            columns = ("stock", "gain", "loss", "prediction", "buy")
            writer = cv.writer(file)
            writer.writerow(columns)     
        
        def add_csv_row(self, stock):
            sp = spm.Stock_Predict(stock)
            gain, loss = sp.get_gain_loss()
            day_new, day_pred, lst_output = sp.predict_future()
            prediction = self.evaluate_prediction(lst_output)
            buy = 1 if (prediction == 1 and gain == 1) else 0 
            print(stock, gain, loss, prediction, buy)
    
            with open("stock.csv", "a", newline="") as f:
                thewriter = cv.writer(f)
                thewriter.writerow([stock, gain, loss, prediction, buy])
                
        def return_row(self,stock):
            sp = spm.Stock_Predict(stock)
            gain, loss = sp.get_gain_loss()
            day_new, day_pred, lst_output = sp.predict_future()
            prediction = self.evaluate_prediction(lst_output)
            buy = 1 if (prediction == 1 and gain == 1) else 0 
            return [gain, loss, prediction]
            
        
        def evaluate_prediction(self,lst):
            a = lst[0][0]
            b = lst[-1][0]
            if b-a>0:
                return 1
            else:
                return 0
        
        def reset_csv(self):
            file = open("stock.csv", "w")
            file.close()
        
        def add_csv_data(self, stocklist):
            self.reset_csv()
            self.make_csv()
            for i in range(len(stocklist)):
                self.add_csv_row(stocklist[i])
                
        def encode_csv(self):
            df = pd.read_csv("stock.csv")
            inputs = df.drop('buy', axis='columns')
            target = df['buy']
            new_inputs = inputs.drop('stock', axis='columns')
            print(new_inputs)
            return new_inputs, target
        
        def read_csv(self):
            df = pd.read_csv("stock.csv")
            print(df.head())
            
class StockDecisionTree():
    
    def __init__(self):
        pass
    
    def making_tree(self):
        from sklearn import tree
        csv = CSV_Maker()
        inputs_encoded, target = csv.encode_csv()
        model = tree.DecisionTreeClassifier()
        model.fit(inputs_encoded, target)
        return model
    
    def make_prediction(self, row:[]):
        ### [gain,loss,prediction] ###
        model = self.making_tree()
        buy = model.predict([row])
        print(buy, type(buy))

# In[4]:


if __name__ == "__main__":
    stocklist = ["AAPL", "MSFT", "SBUX"]
    csv = CSV_Maker()
    csv.add_csv_data(stocklist)
    csv.read_csv()
    csv.encode_csv()
    tree = StockDecisionTree()
    tree.make_prediction(csv.return_row("SNAP"))


# In[ ]:




