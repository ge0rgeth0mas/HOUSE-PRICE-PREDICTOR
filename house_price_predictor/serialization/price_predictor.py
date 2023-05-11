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

# create dictionary of suburb with its property counts and region
suburb_df= X.groupby("suburb")[["propertycount", "regionname"]].agg({'propertycount':'mean', 'regionname':'first'})
# create a series with each uniqiue suburb as index and its corresponding property count and region. Both mean() and size() can be used as each suburb has a fixed propertycount
suburb_dict = suburb_df.to_dict(orient='index') # use the series to create a dictonary

# Remove unnecessary features
features_to_remove = ['suburb', 'address', 'method', 'date', 
                        'postcode', 'bedroom2', 'car', 'landsize',
                        'buildingarea', 'yearbuilt', 'councilarea']
X.drop(features_to_remove, axis=1, inplace=True)

# Change seller names to lowercase
X['sellerg']=X['sellerg'].str.lower()

# Target type, regionname and SellerG features
from category_encoders import TargetEncoder
TargetEncodeCols = ['type', 'regionname', 'sellerg']
te = TargetEncoder(cols=TargetEncodeCols)
X = te.fit_transform(X, y)

X[TargetEncodeCols] /= 100000

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

# Export Target Encoder to pickle file

path = os.getcwd()+'/house_price_predictor/resources/'
te_filename = 'target_encoder.pickle'
try:
    with open(path+te_filename, 'wb') as f:
        pickle.dump(te, f)
    print("Target encoder saved successfully to {}".format(te_filename))

except Exception as e:
    print("Error while saving target encoder to {} : {}".format(te_filename, e))

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

# Export suburb_dict

dictionary_filename = "suburb.json"

try:
    with open(path+dictionary_filename, "w") as f:
        json.dump(suburb_dict, f)
    print("Dictionary saved successfully")
except Exception as e:
    print("Error while saving Dictionary to {} : {}".format(dictionary_filename, e))
