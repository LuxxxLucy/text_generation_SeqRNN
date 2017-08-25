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

import numpy as np

from meta_model.model_session import ModelSession

from pprint import pprint as pr

embed_dim=10

from keras import backend as K
def relative_error(y_true, y_pred):
    '''
    define a customized metric for testing
    '''
    return K.abs(y_true-y_pred) / K.abs(y_true)

class Sequence_RNN_Model_Session(ModelSession):

    def __init__(self,model,args):

        self.model=model
        print('computational graph registered')
        self.args=args
        print('arguments loaded')

        print('session building complete')

    @staticmethod
    def create_graph(class_num=120):

        input_sequences = Input(shape=(40, class_num))
        network=LSTM(49,return_sequences=True)(input_sequences)
        network=LSTM(49,return_sequences=True)(network)
        network=LSTM(49)(network)
        prediction = Dense(class_num, activation='softmax')(network)
        model = Model(inputs=input_sequences, outputs=prediction)
        return model

    @staticmethod
    def compile_model(model):
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy')
        return model

    @classmethod
    def restore(cls, checkpoint_directory,cus=None):
        # custom_objects={'relative_error':relative_error}
        # return super().restore(checkpoint_directory,cus=custom_objects)
        return super().restore(checkpoint_directory)

    def preprocess(self,text):
        maxlen = 40
        self.maxlen=40
        step = 1
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])

        X = np.zeros((len(sentences), maxlen, self.args.class_num),dtype=np.bool)
        y = np.zeros((len(sentences), self.args.class_num),dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char_index in enumerate(sentence):
                X[i, t, char_index] = 1
            y[i, next_chars[i]] = 1

        return X,y

    def train(self,x,y):
        self.model.train_on_batch(x,y)

    def train(self,x):
        x,y=self.preprocess(x)
        self.model.train_on_batch(x,y)

    def evaluate(self,x,batch_size=32):
        x,y=self.preprocess(x)
        result=self.model.evaluate(x,y,batch_size=batch_size,verbose=0)
        return result


    def sample(self,preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate(self,random_sentence_start=None,file_directory=None):

        # sample

        diversity=0.2

        # return list of token
        sentence=[  self.dictionary[it] for it in random_sentence_start[:40]]
        generated=[]

        for i in range(400):
            x = np.zeros((1, self.maxlen, self.args.class_num))
            for t, char_index in enumerate(sentence):
                x[0, t, self.index[char_index]] = 1.

            # iteration get the predictions
            preds = self.model.predict(x, verbose=0)[0]
            next_index = self.sample(preds, diversity)

            next_char = self.dictionary[next_index]

            generated.append(next_char)
            sentence = sentence[1:] + [next_char]

        generated="".join(generated)

        with open(file_directory+"tmp.txt",'w') as f:
            f.write(generated+"\n")

        return generated



    def set_one_hot_depth(self,depth):
        self.one_hot_depth=depth
        print('one hot depth okay')

    def register_dictionary(self,dictionary):
        self.dictionary=dictionary

    def register_index(self,index):
        self.index=index
