import pandas as pd
import numpy as np
import os
from house_price_predictor.data_engineering.loaddata_fromdb import *

# Import Data to Dataframe
melb_house_prices = get_df('melb_data')

# Extract Target column from Data
X = melb_house_prices.drop('price', axis=1)
y = melb_house_prices.price

# Preprocessing

# Convert date to datetime
X['date'] = pd.to_datetime(X['date'], format='%d/%m/%Y')
X['year_sold'] = X['date'].dt.year

# Convert landsize to landsize_log
X['landsize_log'] = pd.Series(np.log(X['landsize']+1))

# Target type, regionname and SellerG features
from category_encoders import TargetEncoder
TargetEncodeCols = ['type', 'regionname', 'sellerg']
te = TargetEncoder(cols=TargetEncodeCols)
X = te.fit_transform(X, y)

X[TargetEncodeCols] /= 100000

# create dictionary of suburb and property counts
suburb_pc_df= X.groupby(['suburb'])['propertycount'].mean() # create a series with each uniqiue suburb as index and its corresponding property count. Both mean() and size() can be used as each suburb has a fixed propertycount
suburb_pc_dict = suburb_pc_df.to_dict() # use the series to create a dictonary


# Remove unnecessary features
features_to_remove = ['suburb', 'address', 'method', 'date', 
                        'postcode', 'bedroom2', 'car', 'landsize',
                        'buildingarea', 'yearbuilt', 'councilarea']
X.drop(features_to_remove, axis=1, inplace=True)

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

# Export column names
import json

columns_filename = "columns.json"
columns = {'data_columns' : [col.lower() for col in X.columns]}

try:
    with open(path+columns_filename, "w") as f:
        f.write(json.dumps(columns))
    print("Coulmns list saved successfully")
except Exception as e:
    print("Error while saving columns to {} : {}".format(columns_filename, e))

# Export suburb_propertycount_dict

dictionary_filename = "propertycount_persuburb.json"

try:
    with open(path+dictionary_filename, "w") as f:
        json.dump(suburb_pc_dict, f)
    print("Dictionary saved successfully")
except Exception as e:
    print("Error while saving Dictionary to {} : {}".format(dictionary_filename, e))