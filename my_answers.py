import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    # From 0 to p-t
    for i in range(len(series) - window_size):  
        inp = []
        
        # Pick next window_size elems
        for j in range(i, i + window_size):
            inp.append(series[j])
            
        # Add input and output
        X.append(inp)
        y.append(series[i+window_size])
        
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)


    # TODO: build an RNN to perform regression on our time series input/output data
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # run your model!
    model.fit(X_train, y_train, epochs=1000, batch_size=50, verbose=0)

    return model

import string

### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    allowed_chars = string.ascii_lowercase + ' ' + '!' + ',' + '.' + ':' + ';' + '?'

    # remove as many non-english characters and character sequences as you can 
    for char in text:
        if char not in allowed_chars:
            text = text.replace(char, '')

    # shorten any extra dead space created above
    text = text.replace('  ',' ')


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # From 0 to p-t
    ctr = 0
    for i in range(window_size, len(text), step_size):
        inputs.append(text[ctr:i])
        outputs.append(text[i])
        ctr = ctr + step_size

    return inputs,outputs
