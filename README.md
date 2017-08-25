# seq-RNN

> still under developing

This is a Python3/ [Keras](https://keras.io) / [Tensorflow](https://www.tensorflow.org/) implementation of paper [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)

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

**I want to train on fewer GPUs**. To train on fewer GPUs we recommend using `CUDA_VISIBLE_DEVICES` to narrow the visibility of GPUs to only a few and then run the script. Don't forget to modulate the flag `--nr_gpu` accordingly.

**I want to train on my own dataset**. Have a look at the `DataLoader` classes in the `data/` folder. You have to write an analogous data iterator object for your own dataset and the code should work well from there.

## usage

indicate the training dataset by

- '--data_set fake_seq':a toy traning set: getting the average value of sequence
- '--data_set linux_data': linux source code (character level)
- '--data_set shakespeare': shakespeare's plays (word-level)

## Pretrained model checkpoint

not yet.

## Citation
