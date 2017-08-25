# char-RNN

> still under developing

This is a Python3/ [Keras](https://keras.io) / [Tensorflow](https://www.tensorflow.org/) implementation of paper [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)

Can use it to do character level text generation and word-level

For character level text generation, I used the linux source code.

For word-level text generation, I used works from shakespeare

## Setup

To run this code you need the following:

- a machine with multiple GPUs
- Python3
- Numpy, Keras

## Training the model

Use the `main_entry.py` script to train the model. To train the default model on CIFAR-10 simply use:

```
python3 main_entry.py
```

## usage

indicate the training dataset by

- '--data_set linux_data': linux source code (character level)
- '--data_set shakespeare': shakespeare's plays (word-level)
