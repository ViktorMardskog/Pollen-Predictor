# Pollen-Predictor
Pollen Predictor

This project is about pollen prediction in Stockholm. Each morning, we ingest:
Max pollen levels for yesterday (for alder pollen, birch pollen, grass pollen, mugwort pollen, olive pollen and ragweed pollen)
A 9 day weather forecast for the following daily variables:
- "temperature_2m_mean",
- "precipitation_sum", 
- "dew_point_2m_mean", 
- "wind_speed_10m_mean", 
- "sunshine_duration", 

The task is to predict the maximum levels of pollen for each pollen type and forecasted day.

We use the following features: 
- doy (day of year, 1–366)
- temperature_2m_mean  --> 7-day rolling mean of temp, 14-day rolling mean of temp
- precipitation_sum --> precip_1d, precip_3d_sum, precip_7d_sum
- dewpoint_2m_mean --> 3-day rolling mean
- wind_speed_10m_mean
- Sunshine Duration

Further, we also use lagged values (pollen levels for t-1, t-2, t-3) to make the predictions (for day t). Autoregressive multi-step forecasting is used.

We have trained one model per pollen type have have measured performance for each model:
Alder pollen model: 

| Metric  | Value |
|---------|-------|
| R²      | 0.8602 |
| MSE_h1  | 5.7022 |
| MSE     | 5.5935 |
| MSE_h2  | 8.4433 |
| MSE_h3  | 7.5775 |
| MSE_h4  | 5.8524 |
| MSE_h5  | 6.5259 |
| MSE_h6  | 7.2237 |
| MSE_h7  | 7.8990 |


