# This script is to be run if the app is to work incase the house_price_predictor.pickle file 
# is not being downloaded from git lfs, as it has early 500MB size

import pandas as pd
import numpy as np
import os

#Load dataframe
data = pd.read_csv(os.getcwd()+"/house_price_predictor/resources/data.csv")

# Get train and target data
X = data.drop('price', axis=1)
y = data['price']

# Train Random Forest Regressor model with n_estimators=450, max_features=4
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=450, max_features=4, random_state=1)
model.fit(X, y)

# Export trained model to pickle file
import pickle

path = os.getcwd()+'/house_price_predictor/resources/'
pickle_filename = 'house_price_predictor.pickle'
try:
    with open(path+pickle_filename, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully to {}".format(pickle_filename))

except Exception as e:
    print("Error while saving model to {} : {}".format(pickle_filename, e))