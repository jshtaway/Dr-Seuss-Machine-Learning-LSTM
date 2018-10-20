import os
import re
import collections
replacements = [('\s+', ' '), ('”', ''), ('“', ''), ('\"', ''), 
                ('\.+', '.'), ('\?', '.'), ('!', '.'), ('-', ' '), (':', ' '), 
                (';', ' '), ('/', ' '), ('\\\\', ' '),("'", ''), ('’\w', ''), 
                ('\(',''), ('\)',''), ('–', ' '), ('‘', ''), ('…',''), ('>',''), (',', '')]

def cleanData(replacements):
    stories = {}
    for root, dirs, files in os.walk('data/sources', topdown=True):
        for name in files: #Only care about the files
            if not re.search('DS_Store', name):
                with open (os.path.join(root,name), 'r+', encoding="utf-8", errors='ignore') as f:
                    stories[name.split('.')[0]] = f.read().lower()#' '.join(f.readlines()).lower()


    with open(os.path.join('.','data','combinedText.txt'), 'w+') as textfile, \
        open(os.path.join('.','data','combinedText.txt'), 'w+') as meta:
        alltext = ''
        for story,text in stories.items():
            for cur, new in replacements:
                print(cur,new)
                text = re.sub(cur, new, text)
            text = text.replace('...','.').replace('..','.')
            meta.write(story+'\n')
            textfile.write(text + ' ')
            alltext += text + ' '
    count = collections.Counter(alltext).most_common()
    print(alltext)
    print([c[0] for c in count])

if __name__ == "__main__":
    cleanData(replacements)