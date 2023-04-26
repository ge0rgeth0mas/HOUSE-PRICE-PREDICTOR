import configparser

config = configparser.ConfigParser()

#create configparser object, and add a section called 'melb_data'

config['melb_data'] = {
    'host' : 'localhost',
    'database' : 'melb_data',
    'username' : 'georgethomas',
    'port' : '5432',
}

with open('dataengineering/mydb_config.ini', 'w') as configfile:
    config.write(configfile)