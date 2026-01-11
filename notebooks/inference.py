from pathlib import Path
import json
import openmeteo_requests
from dotenv import load_dotenv
import datetime
import hopsworks
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
from xgboost import XGBRegressor
from xgboost import plot_importance
import numpy as np

#setUp: 
HOPSWORKS_HOST="c.app.hopsworks.ai"
#May need https://
HOPSWORKS_PROJECT="AirQualityKTHLab1"
city = "stockholm"
pollen_types = ["alder_pollen","birch_pollen","grass_pollen" ,"mugwort_pollen" ,"olive_pollen" ,"ragweed_pollen"]
#Stockholm: 59.334591, 18.063240
#new: 59.338295,18.055603
latitude =59.338295
longitude = 18.055603

today = datetime.date.today()
print("today: " , today)
yesturday =  today - datetime.timedelta(days=1)
print("yesturday: " , yesturday)
root_dir = Path.cwd()
print(root_dir)
load_dotenv()
project = hopsworks.login(host=HOPSWORKS_HOST, project=HOPSWORKS_PROJECT)

fs = project.get_feature_store() 
mr = project.get_model_registry()


#FUNCTION FROM MY LAB1 IN THIS COURSE:
def forecast_recursive(model, yesturday_row, rows_to_predict, feature_cols, max_horizon, use_lags, pollen_type):
    rows = rows_to_predict.sort_values("date").reset_index(drop=True).copy()
    predictions = []
    
    if use_lags: #establish a starting point (today!): 
        lag_1 = yesturday_row[pollen_type]
        lag_2 = yesturday_row[f"{pollen_type}_lag1"]
        lag_3 = yesturday_row[f"{pollen_type}_lag2"]
        
    for h in range(max_horizon): #from today onwards: 
        #row_to_predict = rows_to_predict.iloc[h].copy()
        idx = rows.index[h]

        if use_lags: #overwrite possible real values (which is illegal to use here) with predicted values!
            rows.at[idx, f"{pollen_type}_lag1"] = lag_1
            rows.at[idx, f"{pollen_type}_lag2"] = lag_2
            rows.at[idx, f"{pollen_type}_lag3"] = lag_3
            
        X_row = rows.loc[idx, feature_cols].values.reshape(1, -1)
        y_pred = float(model.predict(X_row)[0])
        y_pred = max(0.0, y_pred)

        predictions.append(y_pred)
        if use_lags:
            #push all lags back because we move forward one day:
            lag_3 = lag_2 
            lag_2 = lag_1
            lag_1 = y_pred

        
    return predictions, rows


#from training.py:
def plot_hindcast(df_hindcast, pollen_type, file_path):
    mapping = {"alder_pollen": 150,"birch_pollen": 1500,"grass_pollen": 40 ,"mugwort_pollen": 30,"olive_pollen": 1 ,"ragweed_pollen": 60}

    date =pd.to_datetime(df_hindcast["date"]).dt.date
    fig, ax =plt.subplots(figsize=(12, 6))

    ax.set_title(f"1-day ahead hindcast for {pollen_type}")
    ax.set_ylabel(f"Pollen level - {pollen_type}")
    ax.set_xlabel("Date")

    ax.plot(
        date,
        df_hindcast[pollen_type],
        marker='^',
        markersize=4,
        label=f"Actual {pollen_type}",
        linewidth=2,
        color='black'
    )
    ax.plot(
        date,
        df_hindcast["predictions"],
        markersize=4,
        label=f"Predicted  {pollen_type}",
        marker='o',
        linewidth=2,
        color='blue'
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.xticks(rotation=45)
    ax.legend(loc="upper left", fontsize="small")
    plt.ylim(0, mapping.get(pollen_type))
    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()

def plot_forcast(df_forcast, pollen_type, file_path):
    mapping = {"alder_pollen": 150,"birch_pollen": 1500,"grass_pollen": 40 ,"mugwort_pollen": 30,"olive_pollen": 1 ,"ragweed_pollen": 60}

    date =pd.to_datetime(df_forcast["date"]).dt.date
    fig, ax =plt.subplots(figsize=(12, 6))

    ax.set_title(f"Forcast for {pollen_type}")
    ax.set_ylabel(f"Pollen level - {pollen_type}")
    ax.set_xlabel("Date")

    ax.plot(
        date,
        df_forcast["predictions"],
        markersize=4,
        label=f"Predicted  {pollen_type}",
        marker='o',
        linewidth=2,
        color='blue'
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.xticks(rotation=45)
    ax.legend(loc="upper left", fontsize="small")
    plt.ylim(0, mapping.get(pollen_type))
    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()


#Some code is taken from the LAB 1 of this course.
def inference(pollen_type):
    print("INFERENCE FOR: ", pollen_type)
    model_name = f"model_{pollen_type}"

    model = mr.get_model(
        name=model_name,
        version=1,
    )

    model_settings = model.download()
    feature_view = model.get_feature_view()

    #we initialize the model in the same way as we did in lab 1:
    XGBoost = XGBRegressor()
    XGBoost.load_model(model_settings + "/model.json") 


    #now im just using the feature view to get the format that matches the training (similarly to my lab1 submission):
    X_example = feature_view.get_batch_data()
    X_example = X_example.drop(columns = ["date"])
    if pollen_type in X_example.columns:
        X_example = X_example.drop(columns =[pollen_type])
    feature_cols =list(X_example.columns)
    print(feature_cols)

    pollen_fg = fs.get_feature_group(
        name='pollen',
        version=5,
    )

    hist = pollen_fg.read()
    hist["date"]  =pd.to_datetime(hist["date"])

    hist = hist.sort_values("date").reset_index(drop=True)

    yesturday_row = hist.iloc[-1]
    
    print(yesturday_row)


    wether_fg = fs.get_feature_group(
        name='weather',
        version=5,
    )

    wether_forecast = wether_fg.filter(wether_fg.date>= yesturday).read().sort_values("date").reset_index(drop=True)       #Want to use the forcast today and onwards to predict!

    rows_to_predict = wether_forecast.copy()
    rows_to_predict[f"{pollen_type}_lag1"] = np.nan
    rows_to_predict[f"{pollen_type}_lag2"] = np.nan
    rows_to_predict[f"{pollen_type}_lag3"] = np.nan
    print(rows_to_predict)

    explected_formats = ['weather_temperature_2m_mean', 'weather_precipitation_sum', 'weather_dew_point_2m_mean', 'weather_wind_speed_10m_mean', 'weather_sunshine_duration', 'weather_doy', 'weather_temp_7d_mean', 'weather_temp_14d_mean', 'weather_precip_3d_sum', 'weather_precip_7d_sum', 'weather_dew_point_3d_mean']
    

    rows_to_predict = rows_to_predict.rename(
        columns={c.replace("weather_", ""): c for c in explected_formats}
    )

    print(rows_to_predict)

    predictions, rows_predicted = forecast_recursive(XGBoost, yesturday_row, rows_to_predict, feature_cols, max_horizon=len(rows_to_predict), use_lags= True, pollen_type = pollen_type)
    print(rows_predicted)
    print(predictions)

    rows_predicted["predictions"] = predictions
    print(predictions)


    rows_predicted["lead_days"] = range(1, len(rows_predicted)+1)
    rows_predicted["pollen_type"] = pollen_type

    rows_predicted = rows_predicted.sort_values("date").reset_index(drop=True) 

    file_path = root_dir / f"pollen_model_{pollen_type}/images/forecast.png"
    plot_forcast(rows_predicted, pollen_type, file_path)

    #monitor: 
    monitoring_data = rows_predicted.copy()
    monitoring_data = monitoring_data[["date", "lead_days" , "pollen_type", "predictions"]]
    predictions_fg = fs.get_or_create_feature_group(
        name = "pollen_predictions",
        description= "Pollen predictions", 
        version = 2,
        primary_key  = ['date','lead_days', 'pollen_type'],
        event_time="date",
    )

    predictions_fg.insert(monitoring_data, wait=True)

    #Now a hindcast:

    monitoring_df = predictions_fg.filter(predictions_fg.lead_days == 1).filter(predictions_fg.pollen_type == pollen_type).read()

    pollen_df = pollen_fg.read()
    pollen_df["date"]  =pd.to_datetime(pollen_df["date"])
    actual_levels_df = pollen_df[["date", pollen_type]]

    df_hindcast = pd.merge(monitoring_df, actual_levels_df, on="date", how="left")
    print(df_hindcast)


    file_path = root_dir / f"pollen_model_{pollen_type}/images/hind_cast.png"

    plot_hindcast(df_hindcast, pollen_type, file_path)
    






def main():
    for pollen_type in pollen_types:
        inference(pollen_type)


if __name__ == "__main__":
    main()

