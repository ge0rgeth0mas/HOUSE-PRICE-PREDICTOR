import os

def get_current_path():
    current_file_path = os.path.abspath(__file__) # get the absolute file path of the current file
    current_folder_path = os.path.dirname(current_file_path) # get the directory path of the current file
    return current_folder_path