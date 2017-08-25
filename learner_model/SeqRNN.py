"""
The core model
"""

import os
from os import path
import time
import settings

import keras
from keras.models import Model
from keras.preprocessing import sequence
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, TimeDistributed
import keras.backend as K

from meta_model.model_session import ModelSession

embed_dim=10

from keras import backend as K
def relative_error(y_true, y_pred):
    '''
    define a customized metric for testing
    '''
    return K.abs(y_true-y_pred) / K.abs(y_true)

class Sequence_RNN_Model_Session(ModelSession):

    @staticmethod
    def create_graph():
        input_sequences = Input(shape=(5, 1))
        network=LSTM(embed_dim)(input_sequences)
        prediction = Dense(1, activation='relu')(network)

        model = Model(inputs=input_sequences, outputs=prediction)

        return model

    @staticmethod
    def compile_model(model):
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=[relative_error])
        return model

    @classmethod
    def restore(cls, checkpoint_directory,cus=None):
        custom_objects={'relative_error':relative_error}
        return super().restore(checkpoint_directory,cus=custom_objects)
