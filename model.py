import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys

filename = "data/combinedText.txt"
raw_text = open(filename).read()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)

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

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filename = "weights-improvement-72-0.9206.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i, c) for i, c in enumerate(chars))

pattern = "the cat in the hat said to the bat that i’m fat. oh no, said the bat, i’m not fat. what say you bird"

x_train = [char_to_int[i] for i in pattern]
x_train = numpy.reshape(x_train, (1,len(x_train),1))
x_train = x_train / float(n_vocab)

for i in range(5000):
    x_train = numpy.reshape(x_train, (1,-1,1))
    prediction = model.predict(x_train, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
#     seq_in = [int_to_char[value] for value in x_train]

    sys.stdout.write(result)
    x_train = numpy.append(x_train, index)
    x_train = x_train[1:len(x_train)]