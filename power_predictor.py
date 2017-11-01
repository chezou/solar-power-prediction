from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

def train_model(_regr, train_dict, target):
  estimators = [('vectorizer', DictVectorizer(sparse=False)), ('regr', _regr)]
  _pl = Pipeline(estimators)
  _pl.fit(train_dict, target)
  
  return _pl

DEFAULT_COLUMNS = ['temperature', 'rain_mm', 'humidity_mbar', 'wind_power',
                   'day_length_sec', 'condition']

def prepare_data(raw_df, predictor_columns=DEFAULT_COLUMNS):
  predictors = raw_df[predictor_columns]
  target = None
  if 'energy' in raw_df.columns:
    target = raw_df.energy

  return predictors, target

def predict_power_generation(_regr, input_df, predictor_columns=DEFAULT_COLUMNS):
  input_dict = prepare_data(input_df, predictor_columns).to_dict(orient='record')
  return _regr.predict(input_dict)
