import numpy as np
import tensorflow as tf
from timit_dataset import TimitDataset
from mock_dataset import MockDataset
from lstm import DeepBiLstmCtc
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# model hyperparameters
batch_size = 64
num_epochs = 200
mfcc_size = 13
lstm_num_hidden = 150
num_layers = 3

def main():
	# load the timit dataset, create a network, train and test it
    dataset = TimitDataset("./data/TIMIT", mfcc_size, split_phonemes=False)
    # dataset = MockDataset(mfcc_size)
    num_cl = dataset.num_classes()  # 61 + blank
    model = DeepBiLstmCtc(mfcc_size, lstm_num_hidden, num_layers, num_cl)
    model.train_and_test(dataset, batch_size, num_epochs)

if __name__ == "__main__":
    main()