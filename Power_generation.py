import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Exploratory analysis before building predictive models

# ## Load CSV

df = pd.read_csv("data/energy.csv")

# Create day length

df['sun_rise_at'] = pd.to_datetime(df['sun__outdoor__rise'], unit='s')
df['sun_set_at'] = pd.to_datetime(df['sun__outdoor__set'], unit='s')


df['day_length'] = df['sun_set_at'] - df['sun_rise_at']
df['day_length_sec'] = df['sun__outdoor__set'] - df['sun__outdoor__rise']

# Show data frame

df

# Rename columns

df2 = df[['datum__bin__date', 'energy__production__inverter', 'temperature__forecast__glen_Dforrest', 'rain__forecast__glen_Dforrest', 'humidity__forecast__glen_Dforrest', 'wind__forecast__glen_Dforrest', 'conditions__forecast__glen_Dforrest', 'day_length', 'day_length_sec']]
df2 = df2.rename(columns={'datum__bin__date': 'date', 'energy__production__inverter': 'energy', 'temperature__forecast__glen_Dforrest': 'temperature', 'rain__forecast__glen_Dforrest': 'rain_mm', 'humidity__forecast__glen_Dforrest': 'humidity_mbar', 'wind__forecast__glen_Dforrest': 'wind_power', 'conditions__forecast__glen_Dforrest': 'condition'})
df2


df2.dtypes

# Plot the pairplot to discover correlation between power generation and other variables.

sns.set(style="ticks")

sns.pairplot(df2, hue="condition")

plt.show()

df2.describe()

# # Build predictive models with linear regression
#
# 1. Split train and test data set
# 2. Filter predictor columns
# 3. Create dummy variables for categorical varibales
# 4. Build model with Linear Regression
# 5. Evaluate
#
# From 1 to 4 should be handle with pipeline model.

# Split development data and test data
# Training data is the range of the first data

dev_data = df2[0:17]
test_data = df2[17:]

def prepare_data(raw_df):
  # Filter columns for predictor
  predictor_columns = ['temperature', 'rain_mm', 'humidity_mbar', 'wind_power', 'day_length_sec', 'condition']

  predictors = raw_df[predictor_columns]
  targets = raw_df.energy

  return predictors, targets
  

energies_train, energies_target = prepare_data(dev_data)

# ## Encode condition from category to dummy variable
from sklearn.feature_extraction import DictVectorizer

vectorizer = DictVectorizer(sparse=False)
energies_cat_train = vectorizer.fit_transform(energies_train.to_dict(orient='record'))


# ## Build a model with linear regression

from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

def rmse(actual, predicted):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    
    return sqrt(mean_squared_error(actual, predicted))


def evaluate_by_loo(energies_train, energies_target, regr=LinearRegression()):

  loo = LeaveOneOut()
  loo.get_n_splits(energies_train)

  train_r2_scores = np.array([])
  test_r2_scores = np.array([])
  train_rmse_scores = np.array([])
  test_rmse_scores = np.array([])
  predicted_powers = np.array([])
  actual_powers = np.array([])

  # Train Linear Regression model
  # It is small data, so 
  for train_index, test_index in loo.split(energies_train):
      print("Test index:{}".format(test_index))
      #print("TRAIN:", train_index, "TEST:", test_index)
      #regr = LinearRegression()

      X_train, X_test = energies_train[train_index], energies_train[test_index]
      y_train, y_test = energies_target.iloc[train_index], energies_target.iloc[test_index]
      regr.fit(X_train, y_train)
      #print(X_test, y_test)
      y_train_pred = regr.predict(X_train)
      y_test_pred = regr.predict(X_test)
      #print(y_test.values, y_test_pred)

      train_r2_score = regr.score(X_train, y_train)
      train_r2_scores = np.append(train_r2_scores, train_r2_score)
      test_r2_score = r2_score(y_test.values, y_test_pred)
      test_r2_scores = np.append(test_r2_scores, test_r2_score)

      train_rmse_score = rmse(y_train, y_train_pred)
      train_rmse_scores = np.append(train_rmse_scores, train_rmse_score)
      test_rmse_score = rmse(y_test.values, y_test_pred)
      test_rmse_scores = np.append(test_rmse_scores, test_rmse_score)

      actual_powers = np.append(actual_powers, y_test.values[0])
      predicted_powers = np.append(predicted_powers, y_test_pred[0])
      print("Actual energy generation: {}\tPredicted energy generation: {}".format(y_test.values[0], y_test_pred[0]))

      print("Train R^2 score: {}\tTest R^2 score:{}".format(train_r2_score, test_r2_score))
      print("Train RMSE: {}\tTest RMSE:{}\n".format(train_rmse_score, test_rmse_score))


  # Standard deviation of training data is base line of RMSE
  print("Standard deviation: {}".format(pd.DataFrame.std(energies_target)))


  print("Train average RMSE: {}\tTest average RMSE:{}".format(np.average(train_rmse_scores), np.average(test_rmse_scores)))
  print("Train average R2: {}\tTest average R2:{}".format(np.average(train_r2_scores), np.average(test_r2_scores)))
  
  return actual_powers, predicted_powers


# Plotting LOO predictions
# http://scikit-learn.org/stable/auto_examples/plot_cv_predict.html
  
actual_powers, predicted_powers = evaluate_by_loo(energies_cat_train, energies_target)
  

def plot_predict_actual(actual, predicted):
  fig, ax = plt.subplots()
  ax.scatter(actual, predicted, edgecolors=(0, 0, 0))
  ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=4)
  ax.set_xlabel('Measured')
  ax.set_ylabel('Predicted')
  plt.show()


plot_predict_actual(actual_powers, predicted_powers)


# Create model with whole data
regr = LinearRegression()
regr.fit(energies_cat_train, energies_target)

energies_test, energies_test_target = prepare_data(test_data)
energies_cat_test = vectorizer.transform(energies_test.to_dict(orient='record'))

energies_pred = regr.predict(energies_cat_test)


print("RMSE: {}\tR2 score: {}".format(rmse(energies_test_target.values, energies_pred), r2_score(energies_test_target.values, energies_pred)))


print('Coefficients: \n', regr.coef_)
print(energies_test.columns)
print('Intercepts: \n', regr.intercept_ )


# We got following formula by linear regression
# 
# $Power\_generation = 608.2 * temperature -5.19 * rain[mm] -301.7 * humidity[mbar] - 344.3 * wind\_power + 2.759 * day\_length[sec] -4802 * condition +  -72793.9$
