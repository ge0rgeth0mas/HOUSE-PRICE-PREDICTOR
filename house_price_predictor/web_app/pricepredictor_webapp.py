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
dictionary_filename = "suburb.json"
try:
    with open(path+dictionary_filename, "r") as f:
        suburb_dict = json.load(f)
    print("Suburb dictionary loaded successfully")
except Exception as e:
    print("Error while loading dictionary from {} : {}".format(dictionary_filename, e))

# Function to predict price
def predict_price(rooms, type, seller, distance, bathroom, lattitude, longtitude, suburb, landsize):
    x = [None] * 11
    x[0]=rooms
    x[1]=type
    x[2]=seller
    x[3]=distance
    x[4]=bathroom
    x[5]=lattitude
    x[6]=longtitude
    x[7]=suburb_dict[suburb]['regionname']
    x[8]=suburb_dict[suburb]['propertycount']
    x[9]=datetime.now().year
    x[10]=float(np.log(landsize+1))
    
    df = pd.DataFrame([x], columns=columns)
    te_df = te.transform(df)
    te_df[['type', 'regionname', 'sellerg']] /=100000

    return te_df, np.round(model.predict(te_df)[0], -2).astype(int)

# Function to return SHAP plots
def shap_gen(df, feat_names):
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

st.write("#### Predict the selling price for your house based on past sales records")

housetype = st.selectbox("Type of property", ["House/Cottage/Villa", "Townhouse", "Unit/Duplex", "Other"], index = 0)
if housetype == "Townhouse":
    type="t"
elif housetype == "Unit/Duplex":
    type="u"
else:
    type="h"

rms, br = st.columns(2)
rooms = rms.number_input("Rooms", min_value=1, max_value=12)
bathroom = br.number_input("Bathrooms", min_value=0, max_value=12)

suburb = st.selectbox("Suburb", list(suburb_dict.keys()), index = 0)
pc, rgn = st.columns(2)
pc.text_input("Properties in suburb",disabled = True, placeholder="{}".format(int(suburb_dict[suburb]['propertycount'])))
rgn.text_input("Region",disabled = True, placeholder="{}".format(suburb_dict[suburb]['regionname']))
lat, lon = st.columns(2)
lattitude = lat.number_input("Latitude", min_value=-38.500000, max_value=-37.000000, value = -37.809203, format = "%.6f")
longtitude = lon.number_input("Longitude", min_value=144.300000, max_value=145.600000, value = 144.995216, format = "%.6f")

distance = st.number_input("Distance from the Melbourne Central Business District (CBD), in km")

landsize = st.number_input("Size of land, in sq.meter", format = "%.3f")

seller = (st.text_input("Enter name of Seller"))


if st.button("Predict price"):
    df, price = predict_price(rooms, type, seller.lower(), distance, bathroom, lattitude, longtitude, suburb, landsize)
    # Output price
    st.success("Predicted selling price for the property is : ${:,}".format(price))
    # Output importance of features
    st.info("The main factors that contributed to the above mentioned prices are shown below. The prices with positive values shown in green served to increase the overall price while the values in red lowered it.")
    feat_names = ["{} rooms".format(rooms), 
                  "{}".format(housetype),
                  "Seller: {}".format(seller),
                  "{}kms from CBD".format(distance),
                  "{} bathrooms".format(bathroom),
                  "Latitude",
                  "Longitude",
                  "{} region".format(suburb_dict[suburb]['regionname']),
                  "{} properties in {}".format(suburb_dict[suburb]['propertycount'], suburb),
                  "Year of sale: {}".format(datetime.now().year),
                  "Landsize"
                  ]
    
    feat_weights = shap_gen(df, feat_names)

    fig, ax = plt.subplots()
    sns.set_style("whitegrid")
    ax = sns.barplot(data=feat_weights, palette=['g' if x >= 0 else 'r' for x in feat_weights.values[0]])
    ax.set_title("What made the house prices go up/down?")
    ax.set_xlabel("Factors that influenced the House price")
    ax.set_ylabel("Impact on house price(SHAP values)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)