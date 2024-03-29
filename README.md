# **House Price Prediction Web App**
## **An end to end Data Science project using regression models**
---
<br>

### **Overview**

This project demonstrates the various stages of a real-world data science project, from data acquisition to product deployment. The process begins with the collection of data, which is then loaded into a database. The data is queried and transformed into a data frame for analysis. Exploratory Data Analysis is then performed, which reveals valuable insights about the data. These insights inform the feature selection, engineering, and transformation process, which prepares the data for model training.

The transformed data is then used to train a baseline model, from which the best model is selected. The selected model is tuned to find the optimum hyperparameters and features, ensuring the highest possible accuracy. Finally, the optimized model is used to create a web app that predicts the selling price of a house based on the selected features.

---

### **Contents**
1. [Overview](#overview)
2. [Tools and Technologies](#tools-and-technologies)
3. [Folder Structure](#folder-structure)
4. [Process Flow Chart](#process-flow-chart)
5. [Data Engineering](#data-engineering)
6. [Data Analysis](#data-analysis)
7. [Modeling](#modeling)
8. [Serialization](#serialization)
9. [Web App](#web-app)
10. [Running the Web App](#running-the-web-app)
11. [Model Explanation](#model-explanation)
12. [Conclusion](#conclusion)
13. [Future Work](#future-work)
14. [References](#references)

---

### **Tools and Technologies**
- Programming Language: Python
- Database: PostgreSQL
- Data Ingestion: Psycopg2
- ORM: SQLAlchemy
- Data Analysis: Pandas, NumPy
- Data Visualization: Matplotlib, Seaborn
- Machine Learning Libraries: Scikit-learn
- Categorical data encoding: Target Encoder
- Hyperparameter tuning: GridSearchCV
- Serialization: Pickle, JSON
- Web Framework: Streamlit
- Model Interpretation: SHAP, Permutation Importance, Partial Dependence Plots

---

### **Folder Structure**
The project folder contains the following directories and files:

- `readme.md`: Contains the project overview and folder structure.
- `pyproject.toml`: Contains information about the project dependencies.
- `.gitignore`: Contains files and directories that are ignored by Git.
- `resources`: Contains the data, models, and other resources required for the project.
    - `webapp_screenshot1.png`: A screenshot of the web app.
    - `webapp_screenshot2.png`: Another screenshot of the web app.
    - `melb_data.csv`: Open-source data used for modeling.
    - `house_price_predictor.pickle`: A trained model for predicting house prices.
    - `target_encoder.pickle`: An encoded version of categorical features used in the model.
    - `columns.json`: A JSON file containing information about the columns used in the model.
    - `suburb.json`: A JSON file containing information about the property count and region for each suburb.
    - `data.csv`: A CSV file containing the data after feature engineering to generate a trained model in case it is not being downloaded from GitHub due to its large size.
- `data_engineering`: Contains scripts for loading data, creating a database, and getting data from the database.
    - `createdb.sh`: A shell script to create a PostgreSQL database.
    - `db_config.py`: Generate a configuration file for the database.
    - `myd_config.ini`: Configuration file generated for the database.   
    - `getdata_from_csv.py`: A script to load data from the CSV file to a DataFrame.
    - `dataingestion_todb.py`: A script to load data from the DataFrame to the database.
    - `current_folder_path.py`: A function that returns the current directory path.
    - `loaddata_fromdb.py`: A script to load data from the database to a DataFrame.
- `data_analysis`: Contains Jupyter notebooks for data exploration and analysis.
    - `EDA.ipynb`: A notebook containing exploratory data analysis.
- `modeling`: Contains Jupyter notebooks for modeling and hyperparameter tuning.
    - `modeling.ipynb`: A notebook containing data preprocessing, model selection, feature selection, and hyperparameter tuning.
- `serialization`: Contains a script to export the trained model, encoder, and column data.
    - `price_predictor.py`: A script to train the model using selected features and export it to the resources folder, along with the encoder user to handle categorical features and also names of columns.
    - `model_generator.py`: A script to generate `house_price_predictor.pickle` in case its is not being downloaded from github due to its large size.
- `web_app`: Contains a script for the web app deployment.
    - `pricepredictor_webapp.py`: A script to create the web app that takes user input and displays the predicted price along with feature importance using SHAP values.
- `model_explanation`: Contains a Jupyter notebook for model explanation and interpretation.
    - `ml_explainability.ipynb`: A notebook that explores the effects of the features in the model using partial dependence plots, permutation importance, and SHAP values.

---

### **Process Flow Chart**

```mermaid
graph TD;
    A[Get data from CSV] --> |Data Ingestion| B[SQL Database];
    B --> C[Load data from database to dataframe];
    C --> D[EDA];
    D --> E[Initial Feature Selection];
    E --> F[Preprocessing];
    F --> G[Model Selection];
    G --> H[Model Iteration];
    H --> I[Feature Engineering]
    H --> J[Hyperparameter Tuning]
    I --> K{Best Model}
    K --> |No| H;
    J --> K
    K --> |Yes| L[Serialization];
    L --> M[Export Columns];
    L --> N[Export Encoder];
    L --> P[Export Trained Model];
    M --> Q{Web App};
    N --> Q;
    P --> Q;
    O[User Input] --> Q;
    Q --> R[Predicted Price];
    Q --> S[Price Explanation];
```
---

### **Data Engineering**
The historical data on sales prices for houses in Melbourne<sup>[1](#ref1)</sup> was collected as a csv file and and loaded into PostgreSQL database using Psycopg2. This has the following advantages:
1. **Performance**: SQL databases are optimized for handling large amounts of structured data. Querying from a database is faster when the data is large.
2. **Data integrity**: SQL databases have built-in mechanisms for ensuring data integrity, such as constraints, indexes, and transactions. These mechanisms can help prevent data errors and inconsistencies, which can be difficult to detect and correct in a large DataFrame.
3. **Scalability**: SQL databases can handle large datasets that may not fit into memory, and can also be scaled up by adding more computing resources or partitioning data across multiple servers. 
4. **Security**: SQL databases offer advanced security features, such as user authentication and access control, to protect sensitive data.
5. **Collaboration**: By storing data in a centralized SQL database, multiple users can access and query the same dataset concurrently.

A script that loads the queried data into a Pandas dataframe using SQLAlchemy has also been created. This script is used for the remaining processes, when the data is required.

---

### **Data Analysis**
Exploratory Data Analysis(EDA) was conducted on the data to gather insights from the data. Visualizations were created using matplotlib and seaborn to get a deeper understanding of the data. This aided in identifying outliers, understanding correlations, exploring feature engineering possibilities, and in identifying features that might be better dropped due to being redundant or providing little to know value.

---

### **Modeling**
#### **Preprocessing**
The insights gathered from EDA enabled us to perform an initial feature selection and feature engineering. The data was then split into a **training set** and **test set**. 
Examples of feature engineering done are:
1. The natural log of the landsize was observed to show a higher correlation with price and was used to replace the landsize feature.
2. The categorical features were encoded with One-Hot Encoding and Target encoding. Target encoding proved to be the better option and was therefore employed in the final model.

---

#### **Model Selection**
The training and test set was used to evaluate the performance of the following supervised learning models from Scikit-Learn:
1. **Liner Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**

Random Forest Regressor proved to be the better choice for the data as it had the lowest **Mean Absolute Error(MAE)** and the highest **R2 Score**.

---

#### **Model Iteration**
The Random Forest Regressor was then used to evaluate various combinations of features using 5 fold **cross-validation**. This aided in **Feature Selection** to get the best predictions and also to decide what inputs would be required from a user to predict a house's price.
The selected features were:
1. The number of rooms
2. The number of bathrooms
3. The type of property (House/Cottage/Villa, Townhouse, Unit/Duplex, or other)
4. The person who sold the house
5. The coordinates of the house (Latitude and Longitude)
6. The region where the house is located
7. The number of properties in the suburb
8. The year of sale
9. Natural log of the landsize
10. Distance of the property from the Central Business District (CBD)

Once the features that aided in getting the best prediction scores were identified, **Hyperparameter Tuning** was done to identify the parameter that gave the best score in Cross-Validation using **GridSearchCV**. The features and hyperparameters were also tested for **Overfitting** using the test set.

The optimum hyperparameters for the Random Forest Regressor identified were:
* `n_estimators=450`
* `max_features=4`

---

### **Serialization**
The entire process of feature selection, feature engineering and model training was carried out on the entire data in a `price_predictor.py` script for production. The script also exported the following to the resources folder for access by the web app:
1. The trained model (`pickle` file)
2. The fitted target encoder (`pickle` file)
3. The names of features used to train the model (`JSON` file)
4. The Number of properties in each suburb along with the region where each suburb falls (`JSON` file)

---

### **Web App**
The front end of the web app is designed using `streamlit`. The User Interface is kept minimalistic and takes in the following details from the user as Input:
1. Type of property
2. Rooms 
3. Bathrooms 
4. Suburb
5. Coordinates of property
6. Distance from CBD
7. Size of land, and
8. Name of Seller

Upon entering the details and clicking the **Predict** button, the script does the following:
1. It uses the imported Suburbs data to check the Property count and Region corresponding to the Suburb.
2. It uses the imported target encoder to transform the type, region and seller name to a numerical value for use by the ML model.
3. It uses the imported column names to create an observations pertaining to the user's house.
4. It uses the imported trained model to predict the selling price for the property with **78% accuracy** as observed with R2 score.
5. It calculates the **SHAP values** for the observation and displays a bar chart that shows the contribution that each feature had, in raising or lowering the price of the house.


#### **Web App Screenshots**
<img src="./house_price_predictor/resources/webapp_screenshot1.png" alt="Screenshot1 of my app" width="500"/><br>

<img id="output" src="./house_price_predictor/resources/webapp_screenshot2.png" alt="Screenshot2 of my app" width="500"/>

<br>

---

### **Running the Web App**
You can run the web app by executing the following commands in terminal or command prompt.

#### **Prerequisites**
1. Before running the web app, you need to have [Python 3](https://www.python.org/downloads/) and [Poetry](https://python-poetry.org/docs/#installation) installed on your system.

You can also install poetry using the following command in terminal or command prompt

```console
pipx install poetry
```

#### **Clone the Repository**
2. Navigate to the directory where you want to clone the repository.
```console
cd <directory>
```
3. Clone the repository by running the following command in the terminal:
```console
git clone https://github.com/ge0rgeth0mas/HOUSE-PRICE-PREDICTOR.git
```

#### **Installing Dependencies**
4. Once you have cloned the repository, navigate to the root directory of the project in a terminal or command prompt.
```console
cd <directory>/HOUSE-PRICE-PREDICTOR
```
5. Use Poetry to install the necessary dependencies for the web app by running the following command:
```console
poetry install
```
This will install all the dependencies specified in the pyproject.toml file.
6. Activate the virtual environment created by Poetry using the `poetry shell` command:
```console
poetry shell
```

#### **Check before running web app**
7. The web app uses a trained model to make predictions that is stored in resources, with the name `house_price_predictor.pickle`. If this file is missing, run the following command to generate it.
```console
python house_price_predictor/serialization/model_generator.py
```

#### **Running the Web App**
8. After activating the virtual environment and ensuring `house_price_predictor.pickle` is present, you can run the web app using the following command:
```console
streamlit run house_price_predictor/web_app/pricepredictor_webapp.py
```
This will launch the app in your default web browser. From there, you can interact with the app and explore its features.

---

### **Model Explanation**
A Jupyter notebook was used to explain the importance of the features from the data in making predictions. Three tools were used for this purpose, namely, `Permutation Importance`, `Partial Dependence Plots`, and `SHAP` Values (an acronym from SHapley Additive exPlanations).

Note: These tools are used after a model has been fit.

1. **Permutation Importance** - Shows us what features most affect predictions.

Here is the output for our data:
| Weight        | Feature        |
| ------------- | -------------- |
| 0.1829 ± 0.0189 | type         |
| 0.1663 ± 0.0213 | rooms        |
| 0.1629 ± 0.0183 | regionname   |
| 0.1626 ± 0.0051 | distance     |
| 0.1506 ± 0.0189 | landsize_log |
| 0.0942 ± 0.0049 | lattitude    |
| 0.0862 ± 0.0043 | longtitude   |
| 0.0638 ± 0.0069 | bathroom     |
| 0.0585 ± 0.0090 | sellerg      |
| 0.0133 ± 0.0026 | propertycount|
| 0.0044 ± 0.0014 | year_sold    |

The features are shown in decreasing order of importance.

2. **Partial Dependence Plots** - Shows us how a feature affects predictions.

<table>
  <tr>
    <td><img src="./house_price_predictor/model_explanation/images/partial_dependence_1.png" alt="Rooms PDD" width="300" height="300"></td>
    <td><img src="./house_price_predictor/model_explanation/images/partial_dependence_3.png" alt="Distance PDD" width="300" height="300"></td>
    <td><img src="./house_price_predictor/model_explanation/images/partial_dependence_4.png" alt="Bathrooms PDD" width="300" height="300"></td>
  </tr>
  <tr>
    <td><img src="./house_price_predictor/model_explanation/images/partial_dependence_5.png" alt="Latitude PDD" width="300" height="300"></td>
    <td><img src="./house_price_predictor/model_explanation/images/partial_dependence_6.png" alt="Longitude PDD" width="300" height="300"></td>
    <td><img src="./house_price_predictor/model_explanation/images/partial_dependence_7.png" alt="Property Count PDD" width="300" height="300"></td>
  </tr>
  <tr>
    <td colspan="1"><img src="./house_price_predictor/model_explanation/images/partial_dependence_8.png" alt="Landsize PDD" width="300" height="300"></td>
    <td colspan="2"><img src="./house_price_predictor/model_explanation/images/partial_dependence_9.png" alt="Coordinates PDD" width="600" height="300"></td>
  </tr>
</table>

We observe that an increase in number of rooms and bathrooms leads to an increase the price of a property, while being further away from the CBD generally tends to lower the property value. We can also see that the larger the landsize the higher the property value. Latitude, Longitude, and property count gives us some interesting insights as well while having a more non-linear relationship with price.

3. **SHAP Values** - Shows us the impact of each feature for a particular prediction.

An example of the SHAP values being used to explain feature importance for a particular set of features used to make a prediction can be seen in the web app screenshot [above](#output).

---

### **Conclusion**
Real world data on past recorded sales price for houses in Melbourne has been used to develop an end-to-end data science project. The project shows the various stages involved such as data acquisition, data cleaning, exploratory data analysis (EDA), feature engineering, modeling, serialization, and web app deployment along with ML explainability.

---

### **Future Work**
Possible future works include:
1. Including more models such as `LightGBM` and `XGBoost` regressors in model selection.
2. Automating data collection to load more data in to the database.
3. Deploying the web app onto a cloud service.

---

### **References**
1. <a id="ref1"></a>: https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot Melbourne housing prices snapshot from Kaggle

---