import os 
import sys 

WORKING_DIR = os.getcwd()

for PATH_DIR in ('Utils'):
    sys.path.append(os.path.join(WORKING_DIR, PATH_DIR))