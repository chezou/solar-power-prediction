# # Predict with all and filtered columns: 'temperature' and 'condition'
# Following processes are expected to be made as a function

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

import os
os.makedirs("models", exist_ok=True)

from power_predictor import DEFAULT_COLUMNS
from power_predictor import prepare_data
from power_predictor import train_model

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

    # from sklearn.externals import joblib
    joblib.dump(pl, os.path.join('models','{}_{}columns_pipeline.pkl'.format(regr_name, len(target_columns))))

    pred = pl.predict(train_cat)
    dev_rmse = rmse(energies_target.values, pred)
    pred_test = pl.predict(energies_test.to_dict(orient='record'))
    test_rmse = rmse(energies_test_target.values, pred_test)
    print(regr_name, dev_rmse, test_rmse)


# You can to load pkl file as follows:
#
# ```py
# pl_loaded = joblib.load(filename)
# target_dict = prepare_data(input_df).to_dict(orient='record')
# pl_loaded.predict(target_dict)
# ```
