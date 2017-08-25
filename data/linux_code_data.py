'''
data laoder for Movie Lens 100K

'''


import numpy as np
import pickle
from os import path
from pathlib import Path
import settings
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.casual import casual_tokenize
from pprint import pprint as pr

DATA_PATH=path.join(settings.DATA_STORE_PATH,"linux_input.txt")

class DataLoader(object):
    """ an object that generates batches of MovieLens_100K data for training """

    def __init__(self, args,kind):
        """
        Initialize the DataLoader
        :param batch_size: is int, number of examples to load at once
        """

        self.args = args
        self.batch_size = args.batch_size
        self.random_state=args.random_state
        self.shuffle=args.shuffle

        self.return_labels=True
        self.p=0

        # now construct data set
        with open(DATA_PATH,'rb') as f:
            text = f.read()
        # determine size of the data set
        word_set= [ it for it in set(text)]

        val_portion=args.val_portion
        if kind=='train':
            self.data=text
            print("The number of all characters is", len(text))
            self.record_num=len(text)
        elif kind=='test':
            print("not available as test. FATAL error, quit now")
            quit()

        self.dictionary=[ it : word_set[it] for it in range(total_num) ]
        self.index=[ word_set[it]:it for it in range(total_num) ]



    def record_num(self):
        return self.record_num

    def dict(self):
        return self.dictionary

    def get_observation_size(self):
        return self.data.shape[1:]

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.random_state.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        self.p += self.batch_size

        return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)
