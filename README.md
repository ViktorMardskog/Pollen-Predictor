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

Grass pollen model: 

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

Birch pollen model:
| Metric  | Value |
|---------|-------|
| R²      | 0.5684 |
| MSE_h1  | 14471 |
| MSE     | 14195 |
| MSE_h2  | 18645 |
| MSE_h3  | 21971 |
| MSE_h4  | 20586 |
| MSE_h5  | 23315 |
| MSE_h6  | 23622 |
| MSE_h7  | 24482 |


Alder pollen model:
| Metric  | Value |
|---------|-------|
| R²      | 0.0989 |
| MSE_h1  | 87.275 |
| MSE     | 85.611 |
| MSE_h2  | 95.338 |
| MSE_h3  | 94.438 |
| MSE_h4  | 96.753 |
| MSE_h5  | 96.258 |
| MSE_h6  | 99.829 |
| MSE_h7  | 100.95 |

Olive pollen model: 
| Metric  | Value |
|---------|-------|
| R²      | 0 |
| MSE_h1  | 0.0001 |
| MSE     | 0.0001 |
| MSE_h2  | 0.0001 |
| MSE_h3  | 0.0001 |
| MSE_h4  | 0.0001 |
| MSE_h5  | 0.0001 |
| MSE_h6  | 0.0001 |
| MSE_h7  | 0.0001 |

Mugwort pollen model: 
| Metric  | Value |
|---------|-------|
| R²      | 0.8818 |
| MSE_h1  | 1.918 |
| MSE     | 1.8814 |
| MSE_h2  | 2.1848 |
| MSE_h3  | 2.7729 |
| MSE_h4  | 2.758 |
| MSE_h5  | 3.7274 |
| MSE_h6  | 3.6399 |
| MSE_h7  | 3.5608 |

Ragweed pollen model:
| Metric  | Value |
|---------|-------|
| R²      | 0.7448 |
| MSE_h1  | 4.5827 |
| MSE     | 4.4953 |
| MSE_h2  | 4.1479 |
| MSE_h3  | 1.7903 |
| MSE_h4  | 2.7327 |
| MSE_h5  | 2.3723 |
| MSE_h6  | 3.1645 |
| MSE_h7  | 6.6614 |



