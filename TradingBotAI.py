#!/usr/bin/env python
# coding: utf-8

# In[1]:



import StockPredictorModule as spm
import DecisionTree as dt
import schedule, time
#Imports
from pyrh import Robinhood
import dotenv
dotenv.load_dotenv()
import os
import importlib.util
import numpy as np
    
def RHlogin():
    import os
    USERNAME = os.getenv("RH_USERNAME")
    PASSWORD = os.getenv("RH_PASSWORD")

    rh = Robinhood(username=USERNAME, password=PASSWORD)
    rh.login()
    return rh

    
def inspect_historicals(rh, stock:str):
    print("Getting Historical Quotes")
     # Get 5 minute graph data for Ford stock
    historical_quotes = rh.get_historical_quotes("TWTR", "5minute", "day")
    print(historical_quotes['results'][0]["historicals"])


def plot_future_predictions(stock:str):
    stock_predict = spm.Stock_Predict(stock)
    stock_predict.plot_future_predictions()
    


# In[ ]:


if __name__ == "__main__":
  
    ###GET CSV FILES TO TRAIN DECISION TREE###
    csvdata = dt.CSV_Maker()
    stocklist = ["NFLX", "AAPL", "BAC", "GE", "F"]    
    csvdata.add_csv_data(stocklist)                      
    csvdata.read_csv()
    csvdata.encode_csv()                                 
    
    ### IDENTIFYING THE SUCCESS AND FAILURE VARIABLES ###
    
    #s = sched.scheduler(time.time, time.sleep)
    
    
    
    def should_buy(stock:str):
        tree = dt.StockDecisionTree()
        buy = tree.make_prediction(csvdata.return_row(stock))
        return buy
    
    def visualize_predictions(stocklist:[]):
        for i in range(len(stocklist)):
            plot_future_predictions(stocklist[i])
    
    visualize_predictions(stocklist)
    
    def get_instrument(stock,rh):
        instrument = rh.instruments(stock)
        return instrument
    
    
    def run_script():
        import numpy as np
        success = np.array([1])
        fail = np.array([0])
        rh = RHlogin()
        #inspect_historicals(rh)
        instrument = get_instrument("F",rh)
        buy = should_buy("F")
        traded = False
    
        if (buy ==  np.array([1])):
            rh.place_buy_order(instrument, 1)
            traded = True
    
        if (buy == np.array([0]) and traded == True):
            rh.place_sell_order(instrument,1)
    

    #s.enter(300, 1, run_script, (s,))
    #s.run()
    schedule.every(300).seconds.do(run_script)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
       
   

    


# In[ ]:




