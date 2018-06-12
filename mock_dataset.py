import python_speech_features as psf
import scipy.io.wavfile as sciwav
import os
import glob
import numpy as np
from dataset import Dataset

class MockDataset(Dataset):

    # phonems # 61 + 1, numbers [0, 61]

    def __init__(self, mfcc_size):
        assert(mfcc_size == 11)
        self.x_train = np.array([
            np.array([[4, 5, 2, 54, 2, 3, 12, 12, 45, 43, 32], [4, 5, 10, 54, 22, 3, 10, 12, 40, 43, 32]]),
            np.array([[3, 5, 33, 54, 11, 11, 11, 44, 43, 22, 23]])
        ])
        self.y_train = np.array([
            np.array([11, 13]),
            np.array([11, 22])
        ])
        # same for text
        self.x_test = np.array([
            np.array([[4, 5, 2, 54, 2, 3, 12, 12, 45, 43, 32], [4, 5, 10, 54, 22, 3, 10, 12, 40, 43, 32]]),
            np.array([[3, 5, 33, 54, 11, 11, 11, 44, 43, 22, 23]])
        ])
        self.y_test = np.array([
            np.array([11, 13]),
            np.array([11, 22])
        ])