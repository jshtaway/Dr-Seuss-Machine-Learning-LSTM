import os
import re
import json

#--- Convert ipynb file to a python file --- --- ---- --- --- ---- 
for filename in os.listdir('.'):
    if re.search('ipnb', filename):
        with open(filename, 'r') as f:
            nb = f.read()
        print(type(nb))
        nb = json.loads(nb)
        print(type(nb))
        with open(filename.split('.')[0]+'.py','w') as wf:
            for cell in nb['cells']:
                wf.write('\n\n')
                if cell['cell_type'] == 'code':
                    wf.write(cell['source'])
                if cell['cell_type'] == 'markdown':
                    for line in cell['source'].split('\n'):
                        wf.write('#' + line + '\n')