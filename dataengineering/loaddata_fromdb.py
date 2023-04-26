import pandas as pd
import sqlalchemy as db
import configparser
import os

#Read config file containing databas login credentials
config=configparser.ConfigParser()
path = os.getcwd()
if __name__ == '__main__':
    path = path+'/dataengineering/mydb_config.ini' #in case the module is being used directly
else:
    if os.path.basename(os.path.normpath(path))== 'HOUSE-PRICE-PREDICTOR':
        path = path+'/dataengineering/mydb_config.ini' # in case the module is being called from a python script that gets CWD from os.getcwd()
    else:
        parent_dir = os.path.dirname(path)
        path = parent_dir+'/dataengineering/mydb_config.ini' # in case the module is being called from an IPYNB that gets its working folder from os.getcwd()

config.read(path)

hostname = config['melb_data']['host'] #['melb_data'] is the section name within the config file.
database = config['melb_data']['database']
username = config['melb_data']['username']
port_id = config['melb_data']['port']
#no password was set

def get_df (tab_name):
    engine = None
    df = pd.DataFrame()
    try:
        engine = db.create_engine(f'postgresql://{username}@{hostname}:{port_id}/{database}')
        
        with engine.connect() as conn:
            query = f'SELECT * FROM {tab_name}'
            df = pd.read_sql(db.text(query), conn)

    except Exception as error:
        print(error)
    finally:
        if engine is not None:
            engine.dispose()
    
    return df