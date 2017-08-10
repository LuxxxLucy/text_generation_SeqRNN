'''
data laoder for Movie Lens 100K

'''


import numpy as np
import pickle
from os import path
from pathlib import Path
from meta_model import settings
import random

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
        # simply calculate the mean value of each history
        X_data=[]
        Y_data=[]

        total_num=2000

        for _ in range(total_num):
            tmp=[]
            # for length in range(random.randint(4,10)):
            for length in range(5):
                tmp.append([random.random()*10])
            y=float(sum([it[0] for it in tmp ])) / max(len(tmp), 4)
            X_data.append(tmp)
            Y_data.append([y])


        val_portion=args.val_portion
        if kind=='train':
            X_data=np.asarray(X_data[:int(total_num*val_portion)])
            Y_data=np.asarray(Y_data[:int(total_num*val_portion)])
        if kind=='test':
            X_data=np.asarray(X_data[int(total_num*val_portion):])
            Y_data=np.asarray(Y_data[int(total_num*val_portion):])



        if not len(X_data) == len(Y_data):
            print("The number of records in X_data and Y_data is not equal!")
            quit()
        print("The number of records is", len(X_data))
        self.record_num=len(X_data)

        self.data=X_data
        self.labels=Y_data


    def record_num(self):
        return self.record_num

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
        y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)
