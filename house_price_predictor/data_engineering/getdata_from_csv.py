import os
import pandas as pd

directory = os.getcwd()+'/resources/'
file_names = os.listdir(directory)

csv_name = ''

for file in file_names:    #assumes there's only one csv file
    if file.endswith(".csv"):
        csv_name = file

def get_data(fname = csv_name, path = directory):
    return csv_name.split('.')[0], pd.read_csv(path+fname)

print(csv_name.split('.')[0])
