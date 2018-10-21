import os
import re
import ast
import json

#--- Convert ipynb file to a python file --- --- ---- --- --- ---- 
for filename in os.listdir('.'):
    if re.search('ipynb', filename):
        print(filename)
        os.system(f'jupyter nbconvert --to script {filename}.ipynb')