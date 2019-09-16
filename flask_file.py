# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import csv, json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from calendar import monthrange
from datetime import datetime, timedelta

app=Flask(__name__)
CORS(app)

def mlwilldoit(time):
    arima_fitted_model = ARIMAResults.load("models/arima_model.pkl")
    diff = monthdelta(datetime.strptime("2017-07-01","%Y-%m-%d"),datetime.strptime(time,"%Y-%m-%d"))
    return str(arima_fitted_model.forecast(diff)[0][-1])

def monthdelta(d1, d2):
    delta = 0
    while True:
        mdays = monthrange(d1.year, d1.month)[1]
        d1 += timedelta(days = mdays)
        if d1 <= d2:
            delta += 1
        else:
            break
    return delta

def csv_to_json(csv_file = "C:\Python35\projs\ml\cpi_prediction\data\CPIndex_Jan13-To-Jul19.csv"):
    df = pd.read_csv(csv_file)
    columns = list(df)
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        out = json.dumps([row for row in reader])
        return out



@app.route("/time/<time>")
def current_value(time):
    return mlwilldoit(time)

@app.route("/getjson")
def get_json():
    return csv_to_json()


if __name__ == "__main__":
	app.run(host = "0.0.0.0", debug = True)
