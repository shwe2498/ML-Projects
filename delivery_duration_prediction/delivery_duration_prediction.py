# -*- coding: utf-8 -*-
"""Delivery Duration Prediction

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1L4HoWhXmoZM52C06C-VAmGK8wqYDRJIQ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""

> Exploring and Understanding the data

"""

# Read data
#historical_data = pd.read_csv('historical_data.csv')
historical_data = pd.read_csv("datasets.zip",compression='zip')
historical_data.head()

historical_data.info()

historical_data['created_at'] = pd.to_datetime(historical_data['created_at'])
historical_data['actual_delivery_time'] = pd.to_datetime(historical_data['actual_delivery_time'])

historical_data.info()

"""> Feature Creation

Target variable = Actual Delivery time - order creation time
"""

# Create the target variable for regression
from datetime import datetime
historical_data['actual_total_delivery_duration'] = (historical_data['actual_delivery_time'] - historical_data['created_at']).dt.total_seconds()

historical_data.info()

"""Available and busy dasher information may be helpful.
Total number of available dashers within a certain area will change from time to time.
Busy dahers ratio % = toatal busy dashers /total onshift dashers
"""

historical_data['busy_dashers_ratio'] = historical_data['total_busy_dashers']/historical_data['total_onshift_dashers']

historical_data.info()

"""Higher the busy dashers ratio, leser the dasher capacity. Hence, delivery will be longer.

1. customer places order (estimated_order_place_duration) ->
2. restaurant recieves order (restaurant_preparation_time) ->
3. restaurant repares meal (estimated_store_to_consumer_driving_duration) ->
4. meal is delivered to consumer
"""

# create new features which might be useful
historical_data['estimated_non_prep_duration'] = historical_data['estimated_store_to_consumer_driving_duration'] + historical_data['estimated_order_place_duration']

historical_data.info()

"""

> Data Preparation for modelling

"""

# Check ids and decide whether to encode or not
historical_data['market_id'].nunique()

historical_data['store_id'].nunique()

historical_data['order_protocol'].nunique()

# Create dummied for order protocol
order_protocol_dummies = pd.get_dummies(historical_data.order_protocol)
order_protocol_dummies = order_protocol_dummies.add_prefix('order_protocol_')

order_protocol_dummies.head()

market_id_dummies = pd.get_dummies(historical_data.market_id)
market_id_dummies = market_id_dummies.add_prefix('market_id_')
market_id_dummies.head()

"""Reference dictionary:
Maps each store_id to most frequent cuisine_categary they have
"""

# create dictionary with most repeated categories of each store to fill null rows where it is possible
#store_id_cuisine = historical_data.groupby('store_id')['cuisine_category'].agg(pd.Series.mode).to_dict()
store_id_unique = historical_data['store_id'].unique().tolist()
store_id_and_category = {store_id:historical_data[historical_data.store_id == store_id].store_primary_category.mode() for store_id in store_id_unique}

def fill(store_id):
  """Return primary store category from the dictionary"""
  try:
    return store_id_and_category[store_id].values[0]
  except:
    return np.nan

# Fill null values
historical_data['nan_free_store_primary_category'] = historical_data.store_id.apply(fill)

# Create dummies for store primary category
store_primary_category_dummies = pd.get_dummies(historical_data.nan_free_store_primary_category)
store_primary_category_dummies = store_primary_category_dummies.add_prefix('category_')
store_primary_category_dummies.head()

historical_data.info()

# Drop created_at, market_id, store_id, store_primary_category, actual_delivery_time, actual_total_delivery_duration
train_df = historical_data.drop(columns =
 ['created_at', 'market_id', 'store_id', 'store_primary_category', 'actual_delivery_time', 'nan_free_store_primary_category', "order_protocol"], axis=1)
train_df.head()

train_df = pd.concat([train_df, order_protocol_dummies, market_id_dummies, store_primary_category_dummies], axis=1)

# Align dtype over dataset
train_df = train_df.astype('float64')
train_df.head()

train_df.describe()

train_df['busy_dashers_ratio'].describe()

# Check infinity balues with using numpy isfinite() funstion
np.where(np.any(~np.isfinite(train_df), axis=0) == True)

# Replace inf values with nan to drop all nans
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop all nans
train_df.dropna(inplace=True)

train_df.shape

"""

> Collinear Features

Collinearity means that the variables are correlated with each other and they have same effects on the model. It makes hard to interpret the model.

Easy way to do is **Correlation Matrix**
A visual representation containing the correlation coefficients between the variable in dataframe.
Use the 'corr' method to create data showing correlation."""

# MASKED
corr = train_df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmp = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmp, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Without MASKED
corr = train_df.corr()

f, ax = plt.subplots(figsize=(11, 9))
cmp = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmp, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

"""As 'category_indonesian' have some proble lets check it...."""

train_df['category_indonesian'].describe()

"""As it have bunch of zero values drop this feature.


Two funstions to test the correlations:
1. Get Redundant values
2. Find top correlation features
"""

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
  """Sort correlations in the descending order and return n highest results"""
  au_corr = df.corr().abs().unstack()
  labels_to_drop = get_redundant_pairs(df)
  au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
  return au_corr[0:n]

print("Top Abosulute Correlations")
print(get_top_abs_correlations(train_df, 20))

# drop created_at, market_id, store_id, store_primary_category, actual_delivery_time, order_protocal
train_df = historical_data.drop(columns =
 ['created_at', 'market_id', 'store_id', 'store_primary_category', 'actual_delivery_time', 'nan_free_store_primary_category', "order_protocol"], axis=1)
# Dont concat market id
train_df = pd.concat([train_df, order_protocol_dummies, store_primary_category_dummies], axis=1)
train_df.head()

# Drop highly correlated features
train_df = train_df.drop(columns=["total_onshift_dashers", "total_busy_dashers", "category_indonesian", "estimated_non_prep_duration"])

# Align dtype over dataset
train_df = train_df.astype('float32')
# Replace inf values with nan to drop all nans
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop all nans
train_df.dropna(inplace=True)

train_df.head()

train_df.shape

print("Top Abosulute Correlations")
print(get_top_abs_correlations(train_df, 20))

# drop created_at, market_id, store_id, store_primary_category, actual_delivery_time, order_protocal
train_df = historical_data.drop(columns =
 ['created_at', 'market_id', 'store_id', 'store_primary_category', 'actual_delivery_time', 'nan_free_store_primary_category', "order_protocol"], axis=1)
# Dont concat market id
train_df = pd.concat([train_df, store_primary_category_dummies], axis=1)

# Drop highly correlated features
train_df = train_df.drop(columns=["total_onshift_dashers", "total_busy_dashers", "category_indonesian", "estimated_non_prep_duration"])

# Align dtype over dataset
train_df = train_df.astype('float32')
# Replace inf values with nan to drop all nans
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop all nans
train_df.dropna(inplace=True)

train_df.head()

print("Top Abosulute Correlations")
print(get_top_abs_correlations(train_df, 20))

"""

> Feature Engineering


Still have some correlated features use **Feature Engineering** - it is a technique that create new variables to simplify the model and increase its accuracy by using the new variables as predictors."""

# New features
train_df['percent_distinct_item_of_total'] = train_df['num_distinct_items'] / train_df['total_items']
train_df['avg_price_per_item'] = train_df['subtotal'] / train_df['total_items']
train_df.drop(columns=['num_distinct_items', 'subtotal'], inplace=True)
print("Top Abosulute Correlations")
print(get_top_abs_correlations(train_df, 20))

train_df['price_range_of_items'] = train_df['max_item_price'] - train_df['min_item_price']
train_df.drop(columns=['max_item_price', 'min_item_price'], inplace=True)
print("Top Abosulute Correlations")
print(get_top_abs_correlations(train_df, 20))

train_df.shape

"""

> Multicollinearilty check

When one predictor variable in multiple regression model can be predicted from the other variables.

VIF (variance infaltion factor) quantifies the severity of multicollinearity.


> Scaler





"""

from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(features):
  """Compute VIF score using variance_inflation_factor()"""
  vif_data = pd.DataFrame()
  vif_data["feature"] = features
  vif_data["VIF"] = [variance_inflation_factor(train_df[features].values, i) for i in range(len(features))]
  return(vif_data.sort_values(by=['VIF']).reset_index(drop=True))

# Apply VIF computation to all columns
features = train_df.drop(columns=['actual_total_delivery_duration']).columns.to_list()
vif_data = compute_vif(features)
vif_data

multicollinearity = True
while multicollinearity:
  hightest_vif_feature = vif_data['feature'].values.tolist()[-1]
  print("I will remove ", hightest_vif_feature)
  features.remove(hightest_vif_feature)
  vif_data = compute_vif(features)
  multicollinearity = False if len(vif_data[vif_data.VIF > 20]) == 0 else True

selected_features = vif_data['feature'].values.tolist()
vif_data

"""

> Feature Selection

PCA

Random Forest with GINI importance to measure the importance of each feature.

"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Selected features are selected in multicollinearity check part
X= train_df[selected_features]
y = train_df['actual_total_delivery_duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_names = [f"feature {i}" for i in range((X.shape[1]))]
forest = RandomForestRegressor(random_state=42)
forest.fit(X_train, y_train)
feats = {} # a dict to hold feature_name: geature_importance
for feature, importance in zip(X, forest.feature_importances_):
    feats[feature] = importance #add the name/value pair
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', figsize=(15, 12))
plt.show()

# Sort and plot the feature importances
importances.sort_values(by='Gini-importance')[-35:].index.tolist()

#PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_Train = X_train.values
X_Train = np.asarray(X_Train)

# Finding normalised array of X_Train
X_std=StandardScaler().fit_transform(X_Train)
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,81,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

"""

> Scalar

Two methods for scaling
- Standard Scalar
- MIN/MAX Scaling

"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Standard Scalar
def scale(scalar, X, y):
  """Apply the selected scalr to features and target variables"""
  X_scalar = scalar
  X_scalar.fit(X=X, y=y)
  X_scaled = X_scalar.transform(X)
  y_scalar = scalar
  y_scalar.fit(y.values.reshape(-1,1))
  y_scaled = y_scalar.transform(y.values.reshape(-1,1))
  return X_scaled, y_scaled, X_scalar, y_scalar

# example to show how to use it
X_scaled, y_scaled, X_scalar, y_scalar = scale(MinMaxScaler(), X, y)

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

from sklearn.metrics import mean_squared_error

def rmse_with_inv_transform(scalar, y_test, y_pred_scaled, model_name):
  """Convert the scaled error to actual error"""
  y_predict = scalar.inverse_transform(y_pred_scaled.reshape(-1,1))
  # return RMSE with squared False
  rmse_error = mean_squared_error(y_test, y_predict[:,0], squared=False)
  print("Error = "'{}'.format(rmse_error)+" in " + model_name)
  return rmse_error, y_predict

"""

> Classical Machine Learning

We will apply 6 different alogorithms helps to find the best performed model.

We will apply 4 different feature set sizes - full, 40, 20 and 10 features(selected by GINI Importance).

We will apply 3 different scalers - standard, Min-Max and No scaler.

We will get 72 results which includes 6 algorithms, 4 feature set sizes and 3 scalers."""

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import linear_model

# create a generic function which can work with multiple machine learning models
def make_regression(X_train, y_train, X_test, y_test, model, model_name, verbose=True):
  """Apply selected regression model to data and measure error"""
  model.fit(X_train, y_train)
  y_predict = model.predict(X_train)
  train_error = mean_squared_error(y_train, y_predict, squared=False)
  y_predict = model.predict(X_test)
  test_error = mean_squared_error(y_test, y_predict, squared=False)
  if verbose:
    print("Train Error = "'{}'.format(train_error)+" in " + model_name)
    print("Test Error = "'{}'.format(test_error)+" in " + model_name)
  trained_model = model
  return trained_model, y_predict, train_error, test_error

pred_dict = {
    "regression_model": [],
    "feature_set": [],
    "scaler": [],
    "RMSE": []
}

regression_models = {
    "Ridge": linear_model.Ridge(),
    "DecisonTree": tree.DecisionTreeRegressor(max_depth=6),
    "RandomForest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "LGBM": LGBMRegressor(),
    "MLP": MLPRegressor(),
    #"Lasso": linear_model.Lasso(),
    #"ElasticNet": linear_model.ElasticNet(),
    #"XGBRegressor": XGBRegressor(),
    #"LGBMRegressor": LGBMRegressor(),
    #"MLPRegressor": MLPRegressor(),
    #"DecisionTreeRegressor": tree.DecisionTreeRegressor(),
    #"SVR": svm.SVR(),
    #"KNeighborsRegressor": neighbors.KNeighborsRegressor(),
    #"LinearRegression": linear_model.LinearRegression()
}

feature_sets = {
    "full_dataset": X.columns.to_list(),
    "selected_features_40": importances.sort_values(by='Gini-importance')[-40:].index.tolist(),
    "selected_features_20": importances.sort_values(by='Gini-importance')[-20:].index.tolist(),
    "selected_features_10": importances.sort_values(by='Gini-importance')[-10:].index.tolist()
}

scalers = {
    "Standard_scaler": StandardScaler(),
    "MinMax_scaler": MinMaxScaler(),
    "NotScale": None
}

# Examine the error for each combination
for feature_set_name in feature_sets.keys():
  feature_set = feature_sets[feature_set_name]
  for scaler_name in scalers.keys():
    print(f"-----scaled with {scaler_name}----- included columns are {feature_set_name}")
    print("")
    for model_name in regression_models.keys():
      if scaler_name == "NotScale":
        X = train_df[feature_set]
        y = train_df['actual_total_delivery_duration']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        make_regression(X_train, y_train, X_test, y_test, regression_models[model_name], model_name, verbose=True)
      else:
        X_scaled, y_scaled, X_scalar, y_scalar = scale(scalers[scaler_name], train_df[feature_set], y)
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
        _, y_predict_scaled, _, _ = make_regression(X_train_scaled, y_train_scaled[:,0], X_test_scaled, y_test_scaled[:,0], regression_models[model_name], model_name)
        rmse_error, y_predict = rmse_with_inv_transform(y_scalar, y_test, y_predict_scaled, model_name)
        pred_dict["regression_model"].append(model_name)
        pred_dict["feature_set"].append(feature_set_name)
        pred_dict["scaler"].append(scaler_name)
        pred_dict["RMSE"].append(rmse_error)

pred_df = pd.DataFrame(pred_dict)
pred_df

pred_df.plot(kind='bar', figsize=(12,8))

"""Let's change the problem by predicting prep_duration and then calculate actual_total_delivery_duration."""

train_df['prep_time'] = train_df['actual_total_delivery_duration'] - train_df['estimated_store_to_consumer_driving_duration']

# not scaling affects the perfomance, so continue to scale but it doesn't matter much with scaler we used
scalers = {
    "Standard_scaler": StandardScaler()
}

feature_sets = {
    "selected_features_40": importances.sort_values(by='Gini-importance')[-40:].index.tolist()
}

regression_models = {
    "LGBM": LGBMRegressor()
}

"""Why drop features here?
Target variable is deroved from these features and collinearity is avoided by dropping them upfront.
"""

for feature_set_name in feature_sets.keys():
  feature_set = feature_sets[feature_set_name]
  for scaler_name in scalers.keys():
    print(f"-----scaled with {scaler_name}----- included columns are {feature_set_name}")
    print("")
    for model_name in regression_models.keys():
      # drop estimated_store_to_consumer_driving_duration and estimated_oder_place_duration
      X = train_df[feature_set].drop(columns=['estimated_store_to_consumer_driving_duration', 'estimated_order_place_duration'])
      y = train_df['prep_time']

      # to get indices
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      train_indices = X_train.index
      test_indices = X_test.index

      #scale
      X_scaled, y_scaled, X_scalar, y_scalar = scale(scalers[scaler_name], X, y)

      # apply indexing
      X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
      _, y_predict_scaled, _, _ = make_regression(X_train_scaled, y_train_scaled[:,0], X_test_scaled, y_test_scaled[:,0], regression_models[model_name], model_name)
      rmse_error, y_predict = rmse_with_inv_transform(y_scalar, y_test, y_predict_scaled, model_name)
      pred_dict["regression_model"].append(model_name)
      pred_dict["feature_set"].append(feature_set_name)
      pred_dict["scaler"].append(scaler_name)
      pred_dict["RMSE"].append(rmse_error)

"""Let's choose the best performing model and extract the pre_duration predictions."""

pred_values_dict = {
    "actual_total_delivery_duration": train_df["actual_total_delivery_duration"][test_indices].values.tolist(),
    "prep_duration_prediction": y_predict[:,0].tolist(),
    "estimated_store_to_consumer_driving_duration": train_df["estimated_store_to_consumer_driving_duration"][test_indices].values.tolist(),
    "estimated_order_place_duration": train_df["estimated_order_place_duration"][test_indices].values.tolist()
}

values_df = pd.DataFrame(pred_values_dict)
values_df

# sum predictions up with non preparation activities such as orderplacing and driving
values_df["sum_total_delivery_duration"] = values_df["prep_duration_prediction"] + values_df["estimated_store_to_consumer_driving_duration"] + values_df["estimated_order_place_duration"]
values_df

# check new error rate
mean_squared_error(values_df["actual_total_delivery_duration"], values_df["sum_total_delivery_duration"], squared=False)

# what if we use another regression to obtain the actual total delivery duration?
X = values_df[['prep_duration_prediction', 'estimated_store_to_consumer_driving_duration', 'estimated_order_place_duration']]
y = values_df['actual_total_delivery_duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regression_models = {
    "LinearRegression": linear_model.LinearRegression(),
    "Ridge": linear_model.Ridge(),
    "DecisonTree": tree.DecisionTreeRegressor(max_depth=6),
    "RandomForest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "LGBM": LGBMRegressor(),
    "MLP": MLPRegressor(),
}

for model_name in regression_models.keys():
  _, y_predict, _, _ = make_regression(X_train, y_train, X_test, y_test, regression_models[model_name], model_name, verbose=False)
  print("RMSE of:", model_name, mean_squared_error(y_test, y_predict, squared=False))

"""

> Deep Learning

"""

import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(42)

# Neural Network
def create_model(feature_set_size):
  #define the mode
  model = Sequential()
  model.add(Dense(16, activation='relu', input_dim=feature_set_size))
  model.add(Dense(1, activation='linear'))

  # compile the model
  rmse = tf.keras.metrics.RootMeanSquaredError()
  model.compile(optimizer='sgd', loss='mse', metrics=[rmse])
  return model

print(f"------scaled with {scaler_name}------ included columns are {feature_set_name}")
print("")
model_name = "ANN"
scaler_name = "Standard_scaler"
X = values_df[['prep_duration_prediction', 'estimated_store_to_consumer_driving_duration', 'estimated_order_place_duration']]
y = values_df['actual_total_delivery_duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_scaled, y_scaled, X_scalar, y_scalar = scale(scalers[scaler_name], X, y)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
print("feature_set_size:", X_train_scaled.shape[1])
model = create_model(feature_set_size=X_train_scaled.shape[1])
history = model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=64, verbose=1)
y_pred = model.predict(X_test_scaled)
rmse_error = rmse_with_inv_transform(y_scalar, y_test, y_pred, model_name)
pred_dict["regression_model"].append(model_name)
pred_dict["feature_set"].append(feature_set_name)
pred_dict["scaler"].append(scaler_name)
pred_dict["RMSE"].append(rmse_error)

"""Epoch means one cycle over the entire data set.
The loss quantifies the differences between actual values and the predicted ones from the entire dataset.
"""

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

"""Officail soultion will be LBGM + Linear regression.
More hyperparameters tunning in ANN can give better result.
"""
