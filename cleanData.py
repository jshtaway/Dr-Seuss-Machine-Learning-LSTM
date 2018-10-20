import os
import re
replacements = [('\n', ''), ('   ', ' '), ('  ', ' '), ('\"', ''), 
                ('\.+', '.'), ('\?', '.'), ('!', '.')]

def cleanData(replacements):
    stories = {}
    for root, dirs, files in os.walk('data/sources', topdown=True):
        for name in files: #Only care about the files
            if not re.search('DS_Store', name):
                with open (os.path.join(root,name), 'r+', encoding="utf-8", errors='ignore') as f:
                    stories[name.split('.')[0]] = ' '.join(f.readlines()).lower()


    with open(os.path.join('.','data','combinedText.txt'), 'w+') as textfile, \
        open(os.path.join('.','data','combinedText.txt'), 'w+') as meta:
        for story,text in stories.items():
            for cur, new in replacements:
                #print(story, ': ', cur,' ',new)
                text = re.sub(cur, new, text)
            meta.write(story+'\n')
            textfile.write(text + ' ')

if __name__ == "__main__":
    cleanData(replacements)