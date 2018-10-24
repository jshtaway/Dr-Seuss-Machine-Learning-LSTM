import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys


# # Training Model

# In[2]:


filename = "data/combinedText.txt"
raw_text = open(filename).read()


# In[6]:


list(set(raw_text))[0]


# In[7]:


# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))


# In[8]:


n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)


# In[9]:

print("9 Initated")
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)


# In[10]:
print('10 initiated')

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


# In[11]:
import datetime
now = datetime.datetime.now()

# define the LSTM model
model = Sequential()
model.add(LSTM(400, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

with open("character_based/info_{now.strftime('%Y-%m-%d_%H-%M')}.txt", 'w+') as f:
    pstr = "{'seq_length': " + str(seq_length) + '}'
    modelstr = """
model = Sequential()
model.add(LSTM(400, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
"""
    f.write(pstr)
    f.write(modelstr)



print("model compiled")
# In[12]:


# define the checkpoint

filepath=f"character_based/wi-{{epoch:02d}}-{{loss:.4f}}_{now.strftime('%Y-%m-%d_%H-%M')}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# In[13]:

print("nuke launching")
history = model.fit(X, y, epochs=500, batch_size=128, callbacks=callbacks_list, verbose=1)
loss_history = history.history
with open("character_based/loss_history_{now.strftime('%Y-%m-%d_%H-%M')}.txt", 'w+') as f:
    f.write(str(loss_history))