#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 23:47:04 2020

@author: neo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 20:27:11 2020

@author: neo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 02:27:58 2020

@author: neo
"""

from random import randint
from numpy import array
from numpy import argmax
#from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense

import numpy as np
import pandas as pd
from glob import glob
from pandas import concat, DataFrame
from keras.models import save_model
from keras.models import load_model

from sklearn import preprocessing
from pickle import dump

from collections import deque

n_steps_in = 20
n_steps_out = 20

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# prepare data for the LSTM
def get_dataset_neo(dataframe, cardinality):
    length = dataframe.shape[0]
    X1, X2, y = deque(maxlen = length), deque(maxlen = length), deque(maxlen = length)
    for index, row in dataframe.iterrows():
        # generate source sequence
        source = np.asarray(row[:n_steps_in])
        # define padded target sequence
        target = np.asarray(row[n_steps_out:])
        target = np.flipud(target)
        # create padded input target sequence
        target_in = np.insert(target[:-1], 0, 0)
        # encode
        src_encoded = to_categorical([source], num_classes=cardinality)
        tar_encoded = to_categorical([target], num_classes=cardinality)
        tar2_encoded = to_categorical([target_in], num_classes=cardinality)
        
        # print(src_encoded.shape)
        # store
        X1.append(src_encoded)
        X2.append(tar2_encoded)
        y.append(tar_encoded)

    return np.array(X1), np.array(X2), np.array(y)

# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model

# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
    return array(output)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]

# VARIABLES HERE!!!------------------------------------------!!!------------------

num_epochs = 200
    
# filename = 'cam_pose_rounded_dataframe_continuous_100000.csv'
filename = 'varied_vel_cam_pose_rounded_dataframe50000.csv'
#-----------------------------------------------------------------------------------------------------------------------------
result = pd.read_csv(filename) 
result = result.reset_index(drop=True)
result = result.drop(columns = ['timestamp'])

result_diff = result.diff(axis=0, periods = 1)

result_diff = result_diff.drop(result_diff.index[0])
# convert to differences of x and differences of y to make to model generic
result = result_diff.reset_index(drop=True)


# result_diff = result_diff.reset_index(drop=True)

#--------------------------------------------------------------------------------------------------------------------
mult_val = 100.0
shift_val = abs(min(result.min(axis = 0)))

result_shift_mult = (result  +  shift_val) * mult_val

x_pos_array = np.array(result_shift_mult.iloc[:,0:1].values)
y_pos_array = np.array(result_shift_mult.iloc[:,1:2].values)

max_x = np.max(x_pos_array) #global
max_y = np.max(y_pos_array)

print(shift_val, max_x, max_y)

#---------------------------------------------------------------------GLOBAL MIN MAX----------------------------------
result = result.iloc[:40000, :] #train on 8000 samples

result_shift_mult = (result  +  shift_val) * mult_val

x_pos_array = np.array(result_shift_mult.iloc[:,0:1].values)
y_pos_array = np.array(result_shift_mult.iloc[:,1:2].values)

#---------------------------------------------------------------------------------------------
pose_x = series_to_supervised(x_pos_array, n_steps_in, n_steps_out)
pose_y = series_to_supervised(y_pos_array, n_steps_in, n_steps_out)

# selector_xy = [True, False] #X = True, Y = False    # IMPORTANT LINE==============================================
selector_xy = [False]
histories = []

for selector in selector_xy:   

    
    encoder_x = 'enc_x_2sig_diff_campose20_exp1.h5' 
    decoder_x = 'dec_x_2sig_diff_campose20_exp1.h5' 
    
    encoder_y = 'enc_y_2sig_diff_campose20_exp1.h5' 
    decoder_y = 'dec_y_2sig_diff_campose20_exp1.h5'  

    
    # configure problem
    var = max_x if selector else max_y
    n_features = int(var) + 1 #check the max value in the dataframe 
    # print(n_features)

    
    # define model
    train, infenc, infdec = define_models(n_features, n_features, 128)
    train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # generate training dataset
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    target_set = pose_x if selector else pose_y
    
    X1, X2, y = get_dataset_neo(target_set, n_features) 
    
    print(X1.shape,X2.shape,y.shape)

    #ad-hoc fix by Neo
    X1 = X1.reshape(X1.shape[0], X1.shape[2], X1.shape[3])
    X2 = X2.reshape(X2.shape[0], X2.shape[2], X2.shape[3])
    y = y.reshape(y.shape[0], y.shape[2], y.shape[3])
    
    # print(X1.shape,X2.shape,y.shape)
    
    # fit model
    history = train.fit([X1, X2], y, epochs= num_epochs, verbose = 1)  #TRAIN    
    histories.append(history)# for plotting

    model_selector_enc = encoder_x if selector else encoder_y #name the model and save
    model_selector_dec = decoder_x if selector else decoder_y
    
    save_model(infenc, model_selector_enc)
    save_model(infdec, model_selector_dec)
    
    #==============================================================================================================='