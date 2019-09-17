# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import csv, json, shutil
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

data_to_send = []

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

def csv_to_json(csv_file = "C:/Python35/projs/ml/cpi_prediction/data/CPIndex_Jan13-To-Jul19.csv"):
    df = pd.read_csv(csv_file)
    columns = list(df)
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        out = json.dumps([row for row in reader])
        return out

#def linear_regression():

def pd_csv_to_json(csv_file = "C:/Python35/projs/ml/cpi_prediction/data/CPIndex_Jan13-To-Jul19.csv"):
    df = pd.read_csv(csv_file)
    df.drop(columns=["Rural", "Urban", "Status"], axis = 1, inplace = True)
    df.rename(columns = {"Combined": "CPI"}, inplace = True)
    df.to_json("data/no_rural_urban_cpi.json", orient = "records")

def train():
    print("Training model now")
    global data_to_send
    cpi_index = pd.read_json("data/no_rural_urban_cpi.json", orient = "records")
    fish_cpi_index = cpi_index[cpi_index.Group==1.0]
    fish_cpi_index = fish_cpi_index[fish_cpi_index["Sub Group"]=="1.1.02"]
    combined_fish_cpi = fish_cpi_index[['Year', 'Month','CPI']]
    months = {"January" : 1, "February" : 2, "March" : 3, "April" : 4, "May": 5, "June" : 6, "July" : 7, "August" : 8, "September" : 9, "October" : 10, "November" : 11, "December" : 12}
    combined_fish_cpi['Month'] = combined_fish_cpi['Month'].map(lambda x: months[x])
    combined_fish_cpi['Timestamp'] = pd.to_datetime({'year' : combined_fish_cpi['Year'], 'month' : combined_fish_cpi['Month'], 'day':[1] * combined_fish_cpi.shape[0]})
    combined_fish_cpi = combined_fish_cpi.drop(["Year", "Month"], axis = 1)
    combined_fish_cpi = combined_fish_cpi.set_index("Timestamp")
    combined_fish_cpi["CPI"] = pd.to_numeric(combined_fish_cpi["CPI"])
    combined_fish_cpi_values = combined_fish_cpi.values
    total_dataset_size = len(combined_fish_cpi.values)
    size = int(len(combined_fish_cpi_values) * 0.7)
    train, test = combined_fish_cpi_values[0:size], combined_fish_cpi_values[size:]
    print("Pre-processing finished")
    history = [x for x in train]
    predictions = []
    print("Commence the real training now")
    for i in range(len(test)):
        sm_model = ARIMA(history,order=(2,1,0))
        sm_model_fit = sm_model.fit(disp=0)
        print(sm_model_fit.summary())
        output = sm_model_fit.forecast()[0]
        predictions.append(output)
        real = test[i]
        history.append(real)
        print('prediction=%f,expected=%f' %(output,real))
        sm_model_fit.save("models/sm_arima_model.pkl")
    actuals = pd.DataFrame(test, index = combined_fish_cpi[int(0.7 * total_dataset_size):].index, columns=['CPI'])
    forecasts = pd.DataFrame(predictions, index = combined_fish_cpi[int(0.7 * total_dataset_size):].index, columns=['predicted_cpi'])
    final_df = actuals.join(forecasts)
    data_to_send = [dict(zip(["month","actual","predicted"],[str(index)[:10],round(i['CPI'], 3),round(i['predicted_cpi'], 3)])) for index,i in final_df.iterrows()]
    with open("data_to_send.json", "w") as f:
        json.dump(data_to_send, f)


@app.route("/time/<time>")
def current_value(time):
    return mlwilldoit(time)

@app.route("/getorigjson")
def get_orig_json():
    shutil.copy("data/no_rural_urban_cpi_original.json", "data/no_rural_urban_cpi.json")
    return jsonify(json.loads(open(r"data/no_rural_urban_cpi_original.json","r").read()))


@app.route("/getjson")
def get_json():
    return open(r"data/no_rural_urban_cpi.json","r").read()

@app.route("/changecpi", methods=['POST'])
def change_json():
    if request.method == 'POST':
        #print(type(request.get_json()))
        content = request.get_json()
        for k,v in content.items():
            if v is None:
                content[k] = ''
        d = pd.read_json("data/no_rural_urban_cpi.json", orient = "records")
        d.fillna('', inplace = True)
        d.loc[(d["Year"] == content["Year"]) & (d["Month"] == content["Month"]) & (d["State"] == content["State"]) & (d["Group"] == content["Group"]) & (d["Sub Group"] == content["Sub Group"]), "CPI"] = content["CPI"]
        d.to_json("data/no_rural_urban_cpi.json", orient = "records")
        return "OK"

@app.route("/train")
def train_model():
    train()
    return open("data_to_send.json", "r").read()


@app.route("/trainedCpiData")
def graph_values():
    return open("data_to_send.json", "r").read()
    

if __name__ == "__main__":
    # train()
    app.run(host = "0.0.0.0", debug = True)
