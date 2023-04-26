import pandas as pd
import psycopg2
import configparser
import getdata_from_csv as csv_df
import subprocess

#Create database using shell command in createdb.sh
subprocess.call("./dataengineering/createdb.sh", shell=True)

#Read config file containing databas login credentials
config=configparser.ConfigParser()
config.read('dataengineering/mydb_config.ini')

hostname = config['melb_data']['host'] #['melb_data'] is the section name within the config file.
database = config['melb_data']['database']
username = config['melb_data']['username']
port_id = config['melb_data']['port']
#no password was set
conn = None #in case connection is not established

#dictionary for converting data types of data frame to corresponding types for database
dtyp_replacments = { 
    'object' : 'varchar', 
    'float64' : 'float', 
    'int64' : 'int',
    'bool' : 'boolean', 
    'datetime64' : 'timestamp',
    'timedelta[ns]' : 'interval'
    }

csv_name, df = csv_df.get_data()

#Make string with column names and data types for data base
tab_col_names = ", ".join("{} {}".format(c, d) for (c,d) in zip(df.columns, df.dtypes.replace(dtyp_replacments))) 

try:
    with psycopg2.connect(
        host = hostname,
        dbname = database,
        user = username,
        port = port_id,
    ) as conn:
        
        
        with conn.cursor() as curr:
            
            curr.execute('DROP TABLE IF EXISTS ' + csv_name)

            #Create table with the right columns
            load_db_script = "CREATE TABLE IF NOT EXISTS %s (%s);" % (csv_name, tab_col_names)
            curr.execute(load_db_script)
            
            #Transfer data from CSV to Table in DB
            csv_file = open('resources/'+csv_name+'.csv') #Open CSV as object in memory
            load_data_script = '''
                                COPY %s FROM STDIN WITH
                                CSV
                                HEADER
                                DELIMITER AS ','
                            '''
            curr.copy_expert(load_data_script % csv_name, csv_file)

            #Check if Data is loaded
            curr.execute('SELECT * FROM %s LIMIT 10;' % (csv_name))
            for record in curr.fetchall():
                print(record[0], record[1], record[2])


except Exception as error:
    print(error)
finally:
    if conn is not None:
        conn.close()