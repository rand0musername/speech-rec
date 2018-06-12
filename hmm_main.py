import numpy as np
from timit_dataset import TimitDataset
from mock_dataset import MockDataset
from ghmm import Ghmm
import os
import warnings

# ignore a deprecation warning in the hmm lib
warnings.filterwarnings("ignore")

# hmm hyperparameters
mfcc_size = 13
n_states = 3
batch_size = 10
n_iter = 100

def main():
	# load the dataset with split phonemes
    dataset = TimitDataset("./data/TIMIT", mfcc_size, split_phonemes=True)
    num_labels = len(TimitDataset.phonemes) - 1
    # create a gaussian hmm, train and test 
    ghmm = Ghmm(mfcc_size=mfcc_size, n_states=n_states, 
                n_iter=n_iter, num_labels=num_labels)
    ghmm.train_and_test(dataset, batch_size=10)

if __name__ == "__main__":
    main()