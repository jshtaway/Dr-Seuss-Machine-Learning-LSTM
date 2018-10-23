
# coding: utf-8

# In[1]:


import numpy, sys, os, pandas as pd
from random import randint
from pickle import dump, load


# In[2]:


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    tokens = text.split()
    print(tokens[:100])
    print('Total Tokens: %d' % len(tokens))
    print('Unique Tokens: %d' % len(set(tokens)))
    return tokens


# In[3]:


# organize into sequences of tokens
#the plus one is because the last val in the list will be the expected prediction. 
#Its our Y-train
def sequencesCreate(length, tokens):
    from keras.preprocessing.text import Tokenizer
    sequences = list()
    for i in range(length, len(tokens)):
        # select sequence of tokens
        seq = tokens[i-length:i]
        # convert into a line
        #line = ' '.join(seq)
        # store
        sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))
    print(f'sequences[0][0]: {sequences[0][0]}')
    
    tokenizer = Tokenizer()
    # integer encode sequences of words
    #sequences = [str(i) for i in sequences]
    # print(f'tokenizer: {tokenizer}')
    tokenizer.fit_on_texts(sequences)
    # print(f'tokenizer: {tokenizer}')
    sequences = tokenizer.texts_to_sequences(sequences)
    # print(f'sequences: {sequences}')
    
    return sequences, tokenizer


# In[15]:


def modelFit(model, modelName, X, y, seq_length, batch_size, epochs):
    from keras.callbacks import ModelCheckpoint
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # define the checkpoint
    filepath=f"wi_{{epoch:02d}}_{{loss:.4f}}_{modelName}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # fit model
    history_callback = model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list)
    return history_callback


# In[54]:


#--- --- ---- --- ---- --- ---- ---- --- ----- ---- ---
# -- Write Files ---- ---- ---- --- ---- --- --- --- -- 
#--- --- ---- --- ---- --- ---- ---- --- ----- ---- ---
def writeFiles(model, modelName, history_callback, modelList, seq_length, total_sequences, epochs, batch_size):
    loss_history = history_callback.history
    
    # save the model to file
    model.save('m_' + modelName + '.h5')
    loss_history['seq_length'] = seq_length
    loss_history['total_sequences'] = total_sequences
    loss_history['batch_size'] = batch_size
    loss_history['epochs'] = epochs
    
    # save losses
    with open('info_' + modelName + '.txt', 'w+') as f:
        f.write(str(modelList))
        f.write('\n')
        f.write(str(loss_history))


# In[6]:


# select a seed text
# seed_text = lines[randint(0,len(lines))]
seed_text = '''Whosever room this is should be ashamed!
His underwear is hanging on the lamp.
His raincoat is there in the overstuffed chair,
And the chair is becoming quite mucky and damp.
His workbook is wedged in the window,
His sweater's been thrown on the floor.
His scarf and one ski are'''

print(seed_text + '\n')


# In[7]:


# define model
def defineModel(vocab_size, seq_length, modelList, length, input_shape):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import LSTM
    from keras.utils import np_utils
    from keras.layers import Embedding, Flatten
    model = Sequential()
    #-- EMBEDDED LAYER --- --- --- ---- --
    #input_dim: size of the vocabulary in the text data.
    #output_dim: size of the vector space where words will be embedded. or size of the output vectors from this layer try 32 or 100 or larger
    #input_length: length of input seq's. ex: if input documents are comprised of 1000 words, it would be 1000.
#     modelList = [{'model':'Embedding', 'input_dim':vocab_size, 'output_dim': 100, 'input_length': seq_length},
#                  {'model': 'LSTM', 'units':256, 'use_bias':True, 'dropout':.2, 'recurrent_dropout': .2}, 
#                  {'model': 'Dense','units':64,'activation':'relu'}, 
#                  {'model': 'LSTM', 'units':256, 'use_bias':True, 'dropout':.2, 'recurrent_dropout': .2}, 
#                  {'model': 'Dense','units':64,'activation':'relu'}, 
#                  {'model':'Flatten'},
#                  {'model': 'Dense','units':vocab_size,'activation':'softmax'},
#                 ]
    for i,layer in enumerate(modelList):
        if layer['model'] == 'Embedding': 
            model.add(Embedding(input_dim=layer['input_dim'], output_dim=layer['output_dim'], 
                                input_length=layer['input_length']))

            print(f"model.add(Embedding(input_dim= {layer['input_dim']}, output_dim={ layer['output_dim'] }, input_length={ layer['input_length'] }))")
        elif layer['model'] == 'LSTM':
            #model.add(LSTM(100, return_sequences=True))
            #model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_dim=1))
            model.add(LSTM(units=layer['units'], use_bias=layer['use_bias'], 
                           dropout=layer['dropout'], recurrent_dropout=layer['recurrent_dropout'], 
                           return_sequences = layer['return_sequences']))
            print(f"model.add(LSTM(units={layer['units']}, use_bias={layer['use_bias']}, dropout={layer['dropout']}, recurrent_dropout={layer['recurrent_dropout']} ))")

        elif layer['model'] == 'Dropout':
            #model.add(Dropout(0.2))
            model.add(Dropout(layer['dropout_rate']))
            print(f"model.add(Dropout({layer['dropout_rate']}))")

        elif layer['model'] == 'Dense':
            #{'model': 'Dense','units':64,'activation':relu'}, 
            #model.add(Dense(100, activation='relu'))
            model.add(Dense(units=layer['units'], activation=layer['activation']))
            print(f"model.add(Dense(units={layer['units']}, activation={layer['activation']}))")

        elif layer['model'] == 'Flatten':
            model.add(Flatten())
            print(f'model.add(Flatten())')
        else:
            raise IOError ('invalid layer')
        
    #Create the model name
    import datetime
    now = datetime.datetime.now()
    modelName = now.strftime("%Y-%m-%d_%H-%M")

    try:
        print(model.summary())
    except:
        pass
    return model, modelName


# In[13]:


def trainModelComplete():
    from keras.utils import to_categorical
    
    #--- PARAMETERS --- --- --- ---- --- --- ---- ---- --- ----- --- --- ----
    #notes from website:
    #-- Common values are 50, 100, and 300. We will use 50 here, --
    #-- but consider testing smaller or larger values. --
    #-- We will use a two LSTM hidden layers with 100 memory cells each. --
    #-- More memory cells and a deeper network may achieve better results. --
    drseuss_text = 'data/combinedText.txt'
    seed_length = 50
    length = seed_length + 1
    epochs = 50
    batch_size = 128
    #-- ---- ---- --- ---- ----- ---- ----- ---- ----- ----- ---- ---- ---- ----
    
    #-- load document --- --- --- --- --
    drseuss_text = 'data/combinedText.txt'
    tokens = load_doc(drseuss_text)

    #-- Create sequencer and tokenizer -- --- --- --- --- --- --- --- 
    sequences, tokenizer = sequencesCreate(length, tokens)
    vocab_size = len(tokenizer.word_index) + 1

    #-- Creating X, y -- --- --- --- --- --- --- -- --
    df = pd.DataFrame(sequences)
    print(f'sequences:\n{df.head(5)}')
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    seq_length = X.shape[1]
    input_shape = X.shape
    #-- One hot encoding -- --- --- --- --- --- -
    y = to_categorical(y, num_classes=vocab_size)
    print(f'seq_length: {seq_length}\nshape of X: {X.shape}\nshape of y: {y.shape}')
    #-- -- ---- --- --- --- --- --- ---- --- --- --- --

    #-- Model List --- --- --- --- --- --- --- --- --- -- ---- --- --- --- ---- -- --
    modelList = [{'model':'Embedding', 'input_dim':vocab_size, 'output_dim': 256, 'input_length': seq_length},
                 {'model': 'LSTM', 'units':256, 'use_bias':True, 'dropout':.2, 'recurrent_dropout': 0, 'return_sequences': True}, 
                 {'model': 'Dense','units':64,'activation':'relu'}, 
                 {'model': 'LSTM', 'units':256, 'use_bias':True, 'dropout':.2, 'recurrent_dropout': 0, 'return_sequences': True}, 
                 {'model': 'Dense','units':64,'activation':'relu'}, 
                 {'model':'Flatten'},
                 {'model': 'Dense','units':vocab_size,'activation':'softmax'},
                ]
#     modelList = [('Embedding', vocab_size, seq_length), ('LSTM',256,'True'), ('Dense',256,'relu'), ('Dropout',.2,''), 
#                  ('LSTM',128,'True'), ('Dense',128,'relu'), ('Dropout',.2,''), 
#                  ('LSTM', 64,'False'), ('Dense',64,'relu'), 
#                  ('Flatten','',''),('Dense',vocab_size,'softmax')]
    #-- --- ---- --- ---- --- --- ---- --- ---- --- ---- --- ---- --- --- --- --- ---
    
    print(f'drseuss_text: \'{drseuss_text}\'\nseed_length: {seed_length}\nepochs: {epochs}\nbatch_size: {batch_size}'
     f'\nmodelList: {modelList}')
    
    #-- Create Model -- --- --- --- ---- --- -- ---- --- --- --- --- --- --- ---- --- ---
    model, modelName = defineModel(vocab_size, seq_length, modelList, length, input_shape)
    #-- save the tokenizer --- --- --- ---- --- --- ---- --
    dump(tokenizer, open('token_'+modelName+'.pkl', 'wb'))
    #-- Fit model -- ---- --- --- --- ---- --- --- ---- --- --- --- --- --- --- --- --- 
    history_callback = modelFit(model, modelName, X, y, seq_length, batch_size, epochs)
    #-- Save history and final model --- -
    writeFiles(model, modelName, history_callback, modelList, seq_length, total_sequences = len(sequences), epochs, batch_size)


# In[22]:


# generate a sequence from a language model
#def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
def generate_seq(modelName, tokenizerName, seq_length, seed_text, n_words):
    from keras.models import load_model
    from keras.preprocessing.sequence import pad_sequences
    # load the model
    model = load_model(modelName)

    # load the tokenizer
    tokenizer = load(open(tokenizerName, 'rb'))
    
    #Make 50 words long
    seed_text = ' '.join(seed_text.split(' ')[0:seq_length])
    
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    
    del model
    return ' '.join(result)


# In[11]:


# modelList = [{'model':'Embedding', 'input_dim':2830, 'output_dim': 256, 'input_length': 50},
#                  {'model': 'LSTM', 'units':256, 'use_bias':True, 'dropout':.2, 'recurrent_dropout': 0, 'return_sequences': True}, 
#                  {'model': 'Dense','units':64,'activation':'relu'}, 
#                  {'model': 'LSTM', 'units':256, 'use_bias':True, 'dropout':.2, 'recurrent_dropout': 0, 'return_sequences': True}, 
#                  {'model': 'Dense','units':64,'activation':'relu'}, 
#                  {'model':'Flatten'},
#                  {'model': 'Dense','units':2830,'activation':'softmax'},
#                 ]
# history_callback = {'history':{'loss': [6.8130, 6.3438, 6.0809, 5.6680, 5.0674, 4.1888, 3.2263, 2.4416, 1.8358, 1.3483, 0.9936, 0.7174, 0.5278, 0.3948, 0.2838, 0.2132, 0.1515, 0.1078, 0.0862, 0.0653, 0.0591, 0.0499, 0.0395, 0.0275, 0.0271, 0.0293, 0.0370, 0.0441, 0.0782, 0.1003, 0.0644, 0.0407, 0.0296, 0.0202, 0.0133, 0.0067, 0.0048, 0.0053, 0.0050, 0.0076, 0.0120, 0.0162, 0.0466, 0.1344, 0.1101, 0.0600, 0.0288, 0.0118, 0.0063], 
#                     'acc':  [0.0366, 0.0477, 0.0514, 0.0527, 0.0647, 0.1239, 0.1239, 0.4201, 0.5374, 0.6495, 0.7304, 0.7957, 0.8472, 0.8845, 0.9168, 0.9394, 0.9593, 0.9714, 0.9791, 0.9854, 0.9866, 0.9900, 0.9918, 0.9948, 0.9946, 0.9945, 0.9913, 0.9897, 0.9782, 0.9701, 0.9821, 0.9881, 0.9925, 0.9952, 0.9975, 0.9991, 0.9997, 0.9996, 0.9996, 0.9987, 0.9984, 0.9962, 0.9854, 0.9570, 0.9661, 0.9805, 0.9917, 0.9974, 0.9993]}}
# writeFiles('NULL', '2018-10-22_11-31', history_callback, modelList, 50, total_sequences = 16175)


# In[16]:


if __name__ == '__main__':
    trainModelComplete()


# In[ ]:


#trainModelComplete()


# In[65]:


def json_create(filepath = '.'):
    import os, ast, json, re, seed
    datetime = {}
    for filename in os.listdir(filepath):
        #wi_01_6.7077__2018-10-22_09-29.hdf5
        m = re.search('wi_(..)_(......)__*(....-..-..)_(..-..).hdf5', filename)
        if m:
            epoch, loss, date, time = m.group(1), m.group(2), m.group(3), m.group(4)
            if date+'_'+time not in datetime.keys():
                #print(f"{date+'_'+time} not in KEYS: \n{datetime.keys()}")
                tokenizer = filepath+f'/token_{date}_{time}.pkl'
                try:
                    with open(filepath+'/info_' + date+'_'+time + '.txt') as f:
                        text = f.read()
                    modelList = text.split(']')[0] + ']'
                    modelHistory = '{' + ']'.join(text.split(']')[1:]).split('{')[1]
                    print(f"NEW DATA: {date+'_'+time}")
                    modelHistory = ast.literal_eval(modelHistory)
                    modelList = ast.literal_eval(modelList)
                    epochs = modelHistory['epochs']
                except:
                    modelList = []
                    modelHistory = {}
                datetime[date+'_'+time] = {'model_list': modelList,
                                           'model_history': modelHistory,
                                           'sequence_list': ['no_model_data']*(epochs+1)}
                print(datetime)
            datetime[date+'_'+time]['sequence_list'][int(epoch)] = generate_seq(os.path.join(filepath,filename), tokenizer, 50, seed.seed_text, 50)
            print('\n',filename, ": ",datetime[date+'_'+time]['sequence_list'][int(epoch)])
    #-- Write JSON file of all model training data --- 
    jsonFile = 'Alldata.json'; i = '0'
    #-- Determine JSON file name -- 
    while os.path.isfile(jsonFile):
        i = str(int(i)+1)
        jsonFile = f"Alldata{i}.json"
    #-- Write JSON file -- --- ----
    with open(jsonFile, 'w+') as fp:
        json.dump(datetime, fp)


# In[66]:


json_create()


# In[ ]:


NEW DATA: 2018-10-22_11-31

 wi_21_0.0591__2018-10-22_11-31.hdf5 :  noise. shouting shouting the place of i at clover that clover again long the i of the never in to ten the no and of the in the the of said while and the of and then of the me and and and but a ie and just the great

 wi_01_6.7077__2018-10-22_09-29.hdf5 :  the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the

 wi_09_1.6076__2018-10-22_09-29.hdf5 :  out now lorax. without without the speck whoville whos whos out with the the they belly three stars from they they you know still of the the had had them them them his belly star who were had bellies things stars. a a a when they in in one the
NEW DATA: 2018-10-22_11-31

 wi_24_0.0369__2018-10-22_11-31.hdf5 :  noise. shouting shouting they place of i the clover they clover that a the they they they you. in to ten the ill all a the in the them the the of think the they thing the and he in in the the will a the they the of the

 wi_02_6.3432__2018-10-22_09-29.hdf5 :  the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
NEW DATA: 2018-10-22_11-31

 wi_14_0.3948__2018-10-22_11-31.hdf5 :  noise. noise. shouting they place up up down down down down than the the they are they are for the who to the they and the boys up in the head while my just just just the of you it it end end end a the grass dog and and
NEW DATA: 2018-10-22_11-31

 wi_16_0.2132__2018-10-22_11-31.hdf5 :  noise. noise. shouting the heard to and down would down clover than long the i then were all in to my the no that his and that and in the i of the of of of then of the in of up you and right this a butter butter heard
NEW DATA: 2018-10-22_11-31

 wi_05_5.0674__2018-10-22_11-31.hdf5 :  noise. noise. the the the the the the the the the the the the the he he he he he he the took took took took took took took took took the the the the the the the the the the the the the the the the the took beast.
NEW DATA: 2018-10-22_11-31

 wi_03_6.0809__2018-10-22_11-31.hdf5 :  the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the

 wi_10_1.1444__2018-10-22_09-29.hdf5 :  out this whole now without my my end my came we your the we house here. just just voice clear clear im pile the of the he would by the shake very were theyd he theyd he sneetches head sing. little feast. feast. worst. worst. worst. worst. worst. worst. worst.
NEW DATA: 2018-10-22_11-31

 wi_13_0.5278__2018-10-22_11-31.hdf5 :  noise. shouting shouting they place up up down down down down and bright hill. the a they raced for to still. the in and a and that up a for head while of your just just who is it of a a a a a can can is i i
NEW DATA: 2018-10-22_11-31

 wi_17_0.1515__2018-10-22_11-31.hdf5 :  noise. noise. shouting the to of and and they that clover that then the i a the and in the and and that that and the in the in the i of the and and that and of the in of the and and a butter butter the with this
NEW DATA: 2018-10-22_11-31

 wi_26_0.0271__2018-10-22_11-31.hdf5 :  noise. noise. shouting the place up i at clover that clover again long the i are the you. for the too to the down and his in and the the side the i just the just the how for to the town ive will a this and the and with

 wi_08_2.2403__2018-10-22_09-29.hdf5 :  out by whole lorax. the of my end my of out again the the just just just just was the of the he still he the he was still still the he a he he packed the he packed the plain grass out out out out out out the they
NEW DATA: 2018-10-22_11-31

 wi_37_0.0067__2018-10-22_11-31.hdf5 :  noise. noise. shouting dont when up his down he air. clover that bright the they are the never for the in the once and a and and and in the the while of just of just who they was be a a and a a make a a and and

 wi_06_4.0345__2018-10-22_09-29.hdf5 :  had the whole the the of they they plain plain plain plain belly belly belly belly belly belly had belly belly belly belly they they sneetches thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars. thars.
NEW DATA: 2018-10-22_11-31

 wi_23_0.0395__2018-10-22_11-31.hdf5 :  noise. noise. shouting the place of i at clover down clover again long he trees said they never in the ten the lifted one the and and the in the head while the what and just then and the of of a all a a a butter the of heard
NEW DATA: 2018-10-22_11-31

 wi_09_1.8358__2018-10-22_11-31.hdf5 :  noise. noise. shouting noise. place up end down down down air. out bright the great great the never for the then the last had the the he he them them how while just just while that that the the end mayor mayor and than the one grass the the morning

 wi_05_4.9483__2018-10-22_09-29.hdf5 :  the the the the the the the the the the the the the knew knew knew their their that they they they they they noise. noise. noise. noise. the the the the plain belly belly the the the the sneetches sneetches sneetches sneetches sneetches sneetches sneetches sneetches sneetches sneetches sneetches
NEW DATA: 2018-10-22_11-31

 wi_15_0.2838__2018-10-22_11-31.hdf5 :  noise. noise. shouting noise. heard up would down down down the that bright the they are were all for to and a and and a up a and how the in thought thought just a they and will the of of of he he as can with the and himself
NEW DATA: 2018-10-22_11-31

 wi_01_6.8130__2018-10-22_11-31.hdf5 :  the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the

 wi_04_5.6446__2018-10-22_09-29.hdf5 :  the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the was was the the
NEW DATA: 2018-10-22_11-31

 wi_07_3.2263__2018-10-22_11-31.hdf5 :  noise. noise. shouting grow. his a a that down the the the the the the the he the im king king king and a a the he them them them them one a and they they they jim jim jim the the the the he them the a a a
NEW DATA: 2018-10-22_11-31

 wi_19_0.0862__2018-10-22_11-31.hdf5 :  noise. noise. shouting the place of and that clover down clover with then the i a the in in the and the in that a and he the in the thought of the i and he then and the of a a and a deep this a all a i
NEW DATA: 2018-10-22_11-31

 wi_18_0.1078__2018-10-22_11-31.hdf5 :  noise. noise. shouting the heard of and down would down clover clover long the i are the i and the in the once more and the that the in the i the you and and they then and the in of the and and then and a the butter a
NEW DATA: 2018-10-22_11-31

 wi_04_5.6680__2018-10-22_11-31.hdf5 :  the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
NEW DATA: 2018-10-22_11-31

 wi_36_0.0133__2018-10-22_11-31.hdf5 :  noise. yertle. shouting dont when of his and he air. clover that again the they are the a for the in the all and a and and and in the great of think the of just the of was in it of you of a a a a a i
NEW DATA: 2018-10-22_11-31

 wi_38_0.0048__2018-10-22_11-31.hdf5 :  noise. yertle. shouting dont when up his down he air. clover that bright the they are the up for the in the all and a and and and in the great while the just of just just just was in a a a a a a a a and i
NEW DATA: 2018-10-22_11-31

 wi_11_0.9936__2018-10-22_11-31.hdf5 :  noise. noise. shouting sad place up would down down down clover again long the of the were never the and then the them and the and that the them the while while the they they high when was the of a the and of the a a the and that
NEW DATA: 2018-10-22_11-31

 wi_22_0.0499__2018-10-22_11-31.hdf5 :  noise. shouting shouting they place of and the clover they clover that a the they are the and in the and and the that a the in the the in the in and the they and the then of a it a the a a a a the a in
NEW DATA: 2018-10-22_11-31

 wi_20_0.0653__2018-10-22_11-31.hdf5 :  noise. shouting shouting the to of i would would down clover and long the i of the and in to too the in down a the in the them of the while i the and and then of have to a a and you a all i just good with
NEW DATA: 2018-10-22_11-31

 wi_12_0.7174__2018-10-22_11-31.hdf5 :  noise. noise. than they they to up down down down down down bright hill. the are they the in and who to once that and and that head a that how your of can can who who is was there there a the to a can can the i and

 wi_07_3.0815__2018-10-22_09-29.hdf5 :  the the the the of put put put the plain plain plain the the had had stars stars all. the the the had had had the the whos whos of of the plain the the of of the when the plain that stars stars stars the the or the the
NEW DATA: 2018-10-22_11-31

 wi_25_0.0275__2018-10-22_11-31.hdf5 :  noise. noise. his the place of i at clover that clover again long he i of the you. in the too was that that a the that up the in the found i the the started for the for the in of will you a this all just the in
NEW DATA: 2018-10-22_11-31

 wi_35_0.0202__2018-10-22_11-31.hdf5 :  noise. noise. shouting dont when up his down he air. clover that bright the they are the never don the in the all and a and and and in the thought while the just of the who of the in a a and he a a a a a and
NEW DATA: 2018-10-22_11-31

 wi_02_6.3438__2018-10-22_11-31.hdf5 :  the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
NEW DATA: 2018-10-22_11-31

 wi_08_2.4416__2018-10-22_11-31.hdf5 :  noise. noise. noise. sad place while up that to down the clover the the trees a the and never to wind the the last the the the his them and head and last while while and and and the he it mayor and than was the to the the to
NEW DATA: 2018-10-22_11-31

 wi_06_4.1888__2018-10-22_11-31.hdf5 :  noise. noise. noise. noise. a a clover clover clover clover the the the the he he he he he he mile splashing splashing the breeze. he he breeze. puzzling splashing trees. a throne. a a air. and and he he the he he he he he he was mile mile
NEW DATA: 2018-10-22_11-31

 wi_10_1.3483__2018-10-22_11-31.hdf5 :  noise. noise. than they place up a down down down clover than the the they a were and for the then and that that and the in of them them the while of your just and that the in of the mayor that the trees the dog and the away.

 wi_03_6.0946__2018-10-22_09-29.hdf5 :  the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the


# In[58]:


['no_model_data']*50

