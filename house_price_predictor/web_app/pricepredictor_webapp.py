import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shap
import seaborn as sns
import os

shap.initjs()
# Import trained model
path = os.getcwd()+'/house_price_predictor/resources/'
import pickle

pickle_filename = 'house_price_predictor.pickle'
try:
    with open(path+pickle_filename, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully from {}".format(pickle_filename))

except Exception as e:
    print("Error while loading model from {} : {}".format(pickle_filename, e))

# Import target encoder
te_filename = 'target_encoder.pickle'
try:
    with open(path+te_filename, 'rb') as f:
        te = pickle.load(f)
    print("Target encoder loaded successfully from {}".format(te_filename))

except Exception as e:
    print("Error while loading target encoder from {} : {}".format(te_filename, e))

# Import column names
import json
columns_filename = "columns.json"
try:
    with open(path+columns_filename, "r") as f:
        columns = json.load(f)['data_columns']
    print("Coulmns list loaded successfully")
except Exception as e:
    print("Error while loading columns from {} : {}".format(columns_filename, e))

# Import suburb_propertycount_dict
dictionary_filename = "propertycount_persuburb.json"
try:
    with open(path+dictionary_filename, "r") as f:
        propertycount_persuburb = json.load(f)
    print("Propertycount dictionary loaded successfully")
except Exception as e:
    print("Error while loading dictionary from {} : {}".format(dictionary_filename, e))


 #["rooms", "type", "sellerg", "distance", "bathroom", "lattitude", "longtitude", "regionname", "propertycount", "year_sold", "landsize_log"]}
# Function to predict price
def predict_price(rooms, type, seller, distance, bathroom, lattitude, longtitude, regionname, landsize):
    x = np.zeros(len(columns))
    x[0]=rooms
    x[1]=distance
    x[2]=bathroom
    x[3]=car
    if yearbuilt==None:
        x[4]=mean_yearbuilt
        x[9]=0
    else:
        x[4]=yearbuilt
        x[9]=1
    x[5]=lattitude
    x[6]=longtitude
    x[7]=datetime.now().year
    x[8]=float(np.log(landsize+1))
    if type == 't':
        x[10]=1
        x[11]=0
    elif type=='u':
        x[10]=0
        x[11]=1
    else:
        x[10]=0
        x[11]=0
    seller = seller.lower()
    if seller in columns:
        seller=seller
    else:
        seller='other'
    pos = columns.index(seller)
    x[pos]=1
    df = pd.DataFrame([x], columns=columns)
    return df, np.round(model.predict(df)[0], -2).astype(int)


# Feature names for shap
rename_feats = {'rooms':"No. of Rooms", 'distance':'Distance from CBD', 'bathroom':'Bathrooms', 'car':'Car parking spots', 'landsize_log': 'landsize', 't':'housetype1', 'u':'housetype2'}
rename_feats.update({columns[i]:'seller{}'.format(i-11) for i in range(12,len(columns))})
feat_names = [rename_feats[val] if val in rename_feats else val for val in columns]

# Function to return SHAP plots
def shap_gen(df):
    #Create shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    
    # Reduce scale of weights
    max_digits = len(
    str(
        abs(shap_values).astype(int).max()
        )
    )
    div = 10**(max_digits-2)
    feat_weights = pd.DataFrame(
                (shap_values/div).astype(int).tolist(), 
                columns=feat_names)
    
    # Select features with higher impact
    feat_weights = feat_weights.loc[:, (feat_weights!=0).any(axis=0)]   # remove features with 0 as value
    return feat_weights

# Web app UI
st.title("Melbourne House Price Predictor")
st.image(path+"melbourne_houses.jpeg")

st.write("#### The web app predicts the selling price of a house based on past data")

st.write("##### Enter the following details to get a prediction of the selling price:")

typ, rms = st.columns([2, 1])
housetype = typ.selectbox("Type of property", ["House/Cottage/villa", "Townhouse", "Unit/Duplex", "Other"], index = 0)
if housetype == "Townhouse":
    type="t"
elif housetype == "Unit/Duplex":
    type="u"
else:
    type="h"
rooms = rms.number_input("Number of rooms in house", min_value=1, max_value=12)
br, cr = st.columns(2)
bathroom = br.number_input("Number of bathrooms in house", min_value=0, max_value=12)
car = cr.number_input("Number of cars parking spots", min_value=0, max_value=12)

yearbuilt = None
year = st.text_input("Enter year in which house was built (Leave blank if information not available) ")
if year=="":
    pass
elif isinstance(year, str):
    try:
        yearbuilt = int(year)
        if yearbuilt<2024 and yearbuilt >1799:
            pass
        else:
            yearbuilt = None
    except Exception as e:
        st.error("Invalid input, error: {}".format(e))
else:
    st.error("Invalid input")

lat, lon = st.columns(2)
lattitude = lat.number_input("House coordinates : Latitude", min_value=-38.500000, max_value=-37.000000, value = -37.809203, format = "%.6f")
longtitude = lon.number_input("House coordinates : Longitude", min_value=144.300000, max_value=145.600000, value = 144.995216, format = "%.6f")

distance = st.number_input("Distance from the Melbourne Central Business DIstrict (CBD), in km")

landsize = st.number_input("Size of land, in sq.meter", format = "%.3f")

seller = st.text_input("Enter name of Seller")

if st.button("Predict price"):
    df, price = predict_price(rooms, distance, bathroom , car, lattitude, longtitude, landsize, type, seller, yearbuilt)
    # Output price
    st.success("Predicted selling price for the property is : ${:,}".format(price))
    # Output importance of features
    st.info("The main features that contributed to the above mentioned prices are show below. The prices with positive values shown in red served to increase the overall price while the values in blue lowered it.")
    feat_weights = shap_gen(df)

    fig, ax = plt.subplots()
    sns.set_style("whitegrid")
    ax = sns.barplot(data=feat_weights, palette=['r' if x >= 0 else 'b' for x in feat_weights.values[0]])
    ax.set_title("Feature Importance")
    ax.set_xlabel("Features")
    ax.set_ylabel("Feature weight")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)