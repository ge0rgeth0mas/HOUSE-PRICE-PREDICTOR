import pandas as pd
import numpy as np
import os
from data_engineering.loaddata_fromdb import *

# Import Data to Dataframe
melb_house_prices = get_df('melb_data')

# Extract Target foring from Data
X = melb_house_prices.drop('price', axis=1)
y = melb_house_prices.price

# Preprocessing

# Convert date to datetime
X['date'] = pd.to_datetime(X['date'], format='%d/%m/%Y')
X['year_sold'] = X['date'].dt.year

# Fill null values in Car Spot
X['car']=X['car'].fillna(0)

# Convert landsize to landsize_log
X['landsize_log'] = pd.Series(np.log(X['landsize']+1))

# Remove data for houses built before 1800
indices_yearbefore1800 = X[X['yearbuilt']<1800].index
X = X.drop(indices_yearbefore1800)
y = y.drop(indices_yearbefore1800)

# Add column to mention which rows have missing yearbuilt
X['yearbuilt_present'] = X['yearbuilt'].notnull().astype(int)
# Fill missing yearbuilt values with mean
mean_yearbuilt = X['yearbuilt'].mean()
X['yearbuilt']=X['yearbuilt'].fillna(mean_yearbuilt)
X['yearbuilt']=X['yearbuilt'].astype(int)

# Change names of sellers with less than 100 sales to 'other'
min_sale = 100
X['sellerg'] = X['sellerg'].apply(lambda x : x.lower() if 
                                              (X['sellerg'].value_counts()[x]>min_sale)
                                              else 
                                              "other")

# One Hot encode type and SellerG features
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
OneHotCols =['type', 'sellerg']
OH_cols= pd.DataFrame(OH_encoder.fit_transform(X[OneHotCols]))
OH_cols.index = X.index # Indices were lost, add it back

OH_col_names = OH_encoder.get_feature_names_out(OneHotCols)
OH_col_names = [s.split('_')[-1] for s in OH_col_names]
OH_cols = OH_cols.rename(columns=dict(zip(OH_cols.columns, OH_col_names)))

X = pd.concat([X, OH_cols], axis=1)

# Remove unnecessary features
features_to_remove = ['suburb', 'address', 'type', 'method', 'sellerg', 'date', 
                      'postcode', 'bedroom2', 'landsize', 'buildingarea', 
                      'councilarea', 'regionname', 'propertycount']
X.drop(features_to_remove, axis=1, inplace=True)

# Train Random Forest Regressor model with n_estimators=375, max_features=21
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=375, max_features=21, random_state=1)
model.fit(X, y)

# Export trained model to pickle file
import pickle

path = os.getcwd()+'/resources/'
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

meanyear_filename = "meanyear.json"
meanyear = {'meanyear' : mean_yearbuilt}

try:
    with open(path+meanyear_filename, "w") as f:
        f.write(json.dumps(meanyear))
    print("Mean year saved successfully")
except Exception as e:
    print("Error while saving mean year to {} : {}".format(meanyear_filename, e))