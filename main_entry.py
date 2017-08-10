# utility modules
import os
from os import path
import shutil
import sys
import time
import json
import argparse
import numpy as np

ITEM_DIM=100

dir_path = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))
sys.path.append(dir_path)

from meta_model import settings
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    # data I/O
    parser.add_argument('--model_directory', type=str, default=settings.MODEL_STORE_PATH,
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('--model_file_name', type=str, default='seq_rnn',
                        help='model file name (will create a separated folder)')
    parser.add_argument('--data_set', type=str, default='fake_seq',
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
    class_num = {'quick_draw': 10,'fake_seq':1}[args.data_set]

    # initialize data loaders for train/test splits
    # data loader

    print(args.data_set)
    if args.data_set == 'quick_draw':
        import data.quick_draw_data as quick_draw_data
        print('start loading dataset',args.data_set)
        dataLoader = quick_draw_data.DataLoader(args)
        print('dataset',args.data_set,'loading completed')
        from learner_model.SketchRNN import Sketch_RNN_Model_Session as model_session
        print('import Sketch RNN model okay')
    elif args.data_set == 'fake_seq':
        import data.fake_seq_data as fake_seq_data
        print('start loading dataset',args.data_set)
        train_data = fake_seq_data.DataLoader(args,'train')
        test_data = fake_seq_data.DataLoader(args,'test')
        print('dataset',args.data_set,'loading completed')
        from learner_model.SeqRNN import Sequence_RNN_Model_Session as model_session
        print('import seq RNN model okay')
    else:
        print('this dataset is not available , or the dataset name not correct')
        quit()


    model_path_name=path.join(args.model_directory,args.model_file_name)
    print(model_path_name)

    if os.path.exists(model_path_name):
        try:
            model = model_session.restore(model_path_name)
        except:
            print("error happens, now remove the original folder name from",model_path_name)
            shutil.rmtree(model_path_name)
            os.makedirs(model_path_name)
            model = model_session.create()
            model.args=args
    else:
        os.makedirs(model_path_name)
        print("There is no model exists in modelpath, so create a new model")
        model = model_session.create()
        model.summary()
    print(model)

    if args.training_num is None:
        args.training_num = train_data.record_num
    print('Last Check :overall training number',train_data.record_num)
    # Train the model, iterating on the data in batches of 32 samples

    iteration=0

    for iEpoch in range(args.training_epoch):
        for x, y in train_data:
            # x, y = training_data.next_batch(args.batch_size)

            # Train the model
            model.train_on_batch(x, y)
            if iteration % args.report_interval == 0:
                score = model.evaluate(x, y, batch_size=args.batch_size)
                print(" training batch score" , score)
            if iteration % args.validation_interval == 0:
                score = model.evaluate(test_data.data, test_data.labels,batch_size=args.batch_size)
                print(" validation score" ,score)
            if iteration % args.checkpoint_interval == 0:
                model_session.save(model,model_path_name)
            iteration+=1
    print("Final model %s" % model)
    model_session.save(model,model_path_name)

    print(" restored from file")
    model_new = model_session.restore(model_path_name)
    score = model_new.evaluate(test_data.data, test_data.labels,batch_size=args.batch_size)
    print(" validation score ",score)
    result=model_new.predict_on_batch(test_data.data[:20])
    print("raw:",test_data.data[:20])
    print("truth:",test_data.labels[:20])
    print("prediction:",result)


def test(args):
    model_path_name=path.join(args.model_directory,args.model_file_name)

    model = ModelSession.restore(model_path_name)
    print(model)
    accuracy = model.test(test_data.X_data, test_data.Y_data)

    print("Test accuracy %0.4f" % accuracy)

if __name__ == "__main__":
    main()
