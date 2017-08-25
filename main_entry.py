# utility modules
import os
from os import path
import shutil
import sys
import time
import json
import argparse
import numpy as np
from pprint import pprint as pr

ITEM_DIM=100

dir_path = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))
sys.path.append(dir_path)

import settings
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    # data I/O
    parser.add_argument('--model_directory', type=str, default=settings.MODEL_STORE_PATH,
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('--model_file_name', type=str, default='seq_rnn',
                        help='model file name (will create a separated folder)')
    parser.add_argument('--data_set', type=str, default='linux_data',
                        help='Can be fake_seq | quick_draw')

    parser.add_argument('--checkpoint_interval', type=int, default=20,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('--report_interval', type=int, default=1,
                        help='Every how many epochs to report current situation?')
    parser.add_argument('--validation_interval', type=int, default=50,
                        help='Every how many epochs to do validation current situation?')
    parser.add_argument('--load_params', dest='load_params', action='store_true',
                        help='Restore training from previous model checkpoint')

    # model
    parser.add_argument('--hist_length', type=int, default=5,
                        help='The minimum length of history sequence')
    parser.add_argument('--training_num', type=int, default=None,
                        help='number of training samples')
    parser.add_argument('--training_epoch', type=int, default=1,
                        help='number of training epoch')
    parser.add_argument('--val_portion', type=float, default=0.4,
                        help='The portion of data to be validation data')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='shuffle the training samples or not')

    # hyper-parameter for optimization
    parser.add_argument('-l', '--learning_rate', type=float,
                        default=0.01, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help='Batch size during training per GPU')
    parser.add_argument('-p', '--dropout_rate', type=float, default=0.2,
                        help='Dropout strength, where 0 = No dropout, higher = more dropout.')
    parser.add_argument('-x', '--max_epochs', type=int, default=5000,
                        help='The maximum epochs to run')
    parser.add_argument('-g', '--nr_gpu', type=int, default=1,
                        help='The number GPUs to distribute the training across')

    # reproducibility:random seed
    parser.add_argument('-s', '--random_seed', type=int, default=42,
                        help='Random seed to use')

    args = parser.parse_args()
    print('INFO CHECK!\ninput args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))


    ################################################
    #        The main program starts
    ################################################

    # fix random seed for reproducibility
    args.random_state = np.random.RandomState(args.random_seed)
    # tf.set_random_seed(args.random_seed)
    train(args)


def train(args):
    class_num = {'quick_draw': 10,'fake_seq':1,'linux_data':101}[args.data_set]
    args.class_num=class_num

    # initialize data loaders for train/test splits
    # data loader

    print(args.data_set)
    if args.data_set == 'linux_data':
        import data.linux_code_data as linux_code_data
        print('start loading dataset',args.data_set)
        train_data = linux_code_data.DataLoader(args,'train')
        print('dataset',args.data_set,'loading completed')
        from learner_model.SeqRNN import Sequence_RNN_Model_Session as model_session
        print('import seq RNN model okay')
    elif args.data_set == 'shakespeare_data':
        import data.shakespeare_data as shakespeare_data
        print('start loading dataset',args.data_set)
        train_data = linux_code_data.DataLoader(args,'train')
        test_data = linux_code_data.DataLoader(args,'test')
        print('dataset',args.data_set,'loading completed')
        from learner_model.SeqRNN import Sequence_RNN_Model_Session as model_session
        print('import seq RNN model okay')
    else:
        print('this dataset is not available , or the dataset name not correct')
        quit()

    model_path_name=path.join(args.model_directory,args.model_file_name)
    print(model_path_name)
    file_path_name=path.join(args.model_directory,args.model_file_name+"Gen")


    if os.path.exists(model_path_name) and args.load_params == True :
        try:
            model = model_session.restore(model_path_name)
        except:
            print("error happens, now remove the original folder name from",model_path_name)
            shutil.rmtree(model_path_name)
            os.makedirs(model_path_name)
            model = model_session.create(class_num=len(train_data.dictionary))
            session = model_session(model,args)
    else:
        try:
            os.makedirs(model_path_name)
        except:
            print("directory okay")

        if os.path.exists(model_path_name) == False:
            print("there is no previous file")
        if args.load_params == False:
            print("deliberately do want to laod a previous model")
        print("create a new model")
        model = model_session.create(class_num=len(train_data.dictionary))
        session = model_session(model,args)
    print(model)

    session.register_dictionary(train_data.dictionary)
    session.register_index(train_data.index)

    if args.training_num is None:
        args.training_num = train_data.record_num
    print('Last Check :overall training number',train_data.record_num)
    # Train the model, iterating on the data in batches of 32 samples

    iteration=0

    for iEpoch in range(args.training_epoch):
        for data in train_data:
            # x, y = training_data.next_batch(args.batch_size)
            x=data
            session.train(x)

            if iteration % args.report_interval == 0:
                score = session.evaluate(data, batch_size=args.batch_size)
                print(" training batch score" , score)
            if iteration % args.validation_interval == 0:
                session.generate(random_sentence_start=x,file_directory=file_path_name)
            if iteration % args.checkpoint_interval == 0:
                session.save(model_path_name)

            iteration+=1
    print("Final model %s" % model)
    model_session.save(model,model_path_name)

def test(args):
    model_path_name=path.join(args.model_directory,args.model_file_name)

    model = ModelSession.restore(model_path_name)
    print(model)
    accuracy = model.test(test_data.X_data, test_data.Y_data)

    print("Test accuracy %0.4f" % accuracy)

if __name__ == "__main__":
    main()
