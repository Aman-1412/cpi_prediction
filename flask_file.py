# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:07:40 2019

@author: Sairam
"""

import pandas as pd
import torch
import torch.nn as nn
from flask import Flask,request
from flask_cors import CORS
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
import numpy as np
from calendar import monthrange
from datetime import datetime, timedelta

app=Flask(__name__)
CORS(app)
#predict
def mlwilldoit(time):
    arima_fitted_model = ARIMAResults.load("models/arima_model.pkl")
    diff = monthdelta(datetime.strptime("2017-07-01","%Y-%m-%d"),datetime.strptime(time,"%Y-%m-%d"))
    return str(arima_fitted_model.forecast(diff)[0][-1])
def ml_forecast_this():
    pass

def monthdelta(d1, d2):
    delta = 0
    while True:
        mdays = monthrange(d1.year, d1.month)[1]
        d1 += timedelta(days=mdays)
        if d1 <= d2:
            delta += 1
        else:
            break
    return delta

@app.route("/time/<time>")
def current_value(time):
    return mlwilldoit(time)

@app.route("/forecast")
def forecast():
    data=request.data['id']
    return ml_forecast_this()


