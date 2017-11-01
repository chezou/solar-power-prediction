# # Predict with all and filtered columns: 'temperature' and 'condition'
# Following processes are expected to be made as a function

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

import os
os.makedirs("models", exist_ok=True)

from power_predictor import DEFAULT_COLUMNS
from power_predictor import prepare_data
from power_predictor import train_model
from power_predictor import predict_power_generation
from power_predictor import rmse

import pandas as pd


# Load CSV

df = pd.read_csv("data/energy.csv")

# Create day length

df['sun_rise_at'] = pd.to_datetime(df['sun__outdoor__rise'], unit='s')
df['sun_set_at'] = pd.to_datetime(df['sun__outdoor__set'], unit='s')

df['day_length'] = df['sun_set_at'] - df['sun_rise_at']
df['day_length_sec'] = df['sun__outdoor__set'] - df['sun__outdoor__rise']

# Rename columns

df2 = df[['datum__bin__date', 'energy__production__inverter', 'temperature__forecast__glen_Dforrest', 'rain__forecast__glen_Dforrest', 'humidity__forecast__glen_Dforrest', 'wind__forecast__glen_Dforrest', 'conditions__forecast__glen_Dforrest', 'day_length', 'day_length_sec']]
df2 = df2.rename(columns={'datum__bin__date': 'date', 'energy__production__inverter': 'energy', 'temperature__forecast__glen_Dforrest': 'temperature', 'rain__forecast__glen_Dforrest': 'rain_mm', 'humidity__forecast__glen_Dforrest': 'humidity_mbar', 'wind__forecast__glen_Dforrest': 'wind_power', 'conditions__forecast__glen_Dforrest': 'condition'})

# Split data
dev_data = df2[0:20]
test_data = df2[20:]

from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV, LassoCV

for target_columns in [DEFAULT_COLUMNS, ['temperature', 'condition']]:
  print(len(target_columns))
  energies_train, energies_target = prepare_data(dev_data, predictor_columns=target_columns)
  energies_test, energies_test_target = prepare_data(test_data, predictor_columns=target_columns)
  print(energies_train.columns)

  print("Name DevRMSE TestRMSE")
  for _regr in [LinearRegression(), ElasticNetCV(cv=4), RidgeCV(cv=4), LassoCV(cv=4)]:
    regr_name = type(_regr).__name__
    train_cat = energies_train.to_dict(orient='record')
    pl = train_model(_regr, train_cat, energies_target)

    # Save model
    joblib.dump(pl, os.path.join('models','{}_{}columns_pipeline.pkl'.format(regr_name, len(target_columns))))

    pred = predict_power_generation(pl, dev_data, predictor_columns=target_columns)
    dev_rmse = rmse(energies_target.values, pred)
    pred_test = predict_power_generation(pl, test_data, predictor_columns=target_columns)
    test_rmse = rmse(energies_test_target.values, pred_test)
    print(regr_name, dev_rmse, test_rmse)


# You can to load pkl file as follows:
#
# ```py
# pl_loaded = joblib.load(filename)
# target, _ = prepare_data(input_df)
# target_dict = target.to_dict(orient='record')
# pl_loaded.predict(target_dict)
# ```
