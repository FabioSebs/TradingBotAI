FROM python:3.8-slim-buster

ADD DecisionTree.py .
ADD StockPredictorModule.py .
ADD TradingBotAI.py . 

RUN pip install python-dotenv numpy matplotlib scikit-learn pandas pandas-datareader tensorflow schedule pyrh

CMD [ "python", "/TradingBotAI.py" ]


