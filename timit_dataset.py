import python_speech_features as psf
import scipy.io.wavfile as sciwav
import os
import glob
import numpy as np
from dataset import Dataset

class TimitDataset(Dataset):

    phonemes = ['h#', 'sh', 'ix', 'hv', 'eh', 'dcl', 'jh', 'ih', 'd', 'ah', 
               'kcl', 'k', 's', 'ux', 'q', 'en', 'gcl', 'g', 'r', 'w', 
               'ao', 'epi', 'dx', 'axr', 'l', 'y', 'uh', 'n', 'ae', 'm', 
               'oy', 'ax', 'dh', 'tcl', 'iy', 'v', 'f', 't', 'pcl', 'ow', 
               'hh', 'ch', 'bcl', 'b', 'aa', 'em', 'ng', 'ay', 'th', 'ax-h', 
               'ey', 'p', 'aw', 'er', 'nx', 'z', 'el', 'uw', 'pau', 'zh', 
               'eng', 'BLANK'] # 61 + 1, numbers [0, 61]

    def __init__(self, timit_root, mfcc_size, split_phonemes):
        self.mfcc_size = mfcc_size
        # load the dataset
        training_root = os.path.join(timit_root, 'TRAIN')
        test_root = os.path.join(timit_root, 'TEST')
        if split_phonemes:
            self.x_train, self.y_train = self.load_split_timit_data(training_root)
            self.x_test, self.y_test = self.load_split_timit_data(test_root)
            self.normalize_xs()
            print(self.x_train.shape) # num_examples=142910 x [num_windows?, mfcc_size=13]
            print(self.y_train.shape) # num_examples=142910 x phonem
            print(self.x_test.shape) # num_examples=51681 x [num_windows?, mfcc_size=13]
            print(self.y_test.shape) # num_examples=51681 x phonem
        else:
            self.x_train, self.y_train = self.load_timit_data(training_root)
            self.x_test, self.y_test = self.load_timit_data(test_root)
            self.normalize_xs()
            print(self.x_train.shape) # num_examples=4620 x [num_windows?, mfcc_size=13]
            print(self.y_train.shape) # num_examples=4620 x [num_phonemes?]
            print(self.x_test.shape) # num_examples=1680 x [num_windows?, mfcc_size=13]
            print(self.y_test.shape) # num_examples=1680 x [num_phonemes?]
            # actually 3696 and 1344 when we drop SA

    def num_classes(self):
        return len(self.phonemes)

    def normalize_xs(self):
        all_xs = np.concatenate([self.x_train, self.x_test])
        all_xs = np.vstack(all_xs.flat)
        mean = np.mean(all_xs)
        std = np.std(all_xs)
        print(mean)
        print(std)
        self.x_train = (self.x_train - mean) / std
        self.x_test = (self.x_test - mean) / std


    def load_split_timit_data(self, root_dir):
        x_list = []
        y_list = []
        wav_glob = os.path.join(root_dir, '**/*.wav')
        it = 1
        for wav_filename in glob.glob(wav_glob, recursive=True):
            if wav_filename[-7:] in ['SA1.wav', 'SA2.wav']:
                # drop SA sentences
                continue
            # load audio
            sample_rate, wav = sciwav.read(wav_filename)
            
            # parse the text file with phonemes
            phn_filename = wav_filename[:-3] + 'PHN' # fragile, i know
            with open(phn_filename) as f:
                lines = f.readlines()
                phonemes = [line.split() for line in lines]

            # slice the wav file and pair up with the corresponding phoneme
            for l, r, ph in phonemes:
                # add x
                wav_slice = wav[int(l) : (int(r)+1)]
                mfcc_data = psf.mfcc(wav_slice, samplerate=sample_rate, numcep=self.mfcc_size,
                                     winlen=0.0125, winstep=0.005)
                x_list.append(mfcc_data)
                if len(x_list) % 100 == 0:
                    print('Added {} pairs.'.format(len(x_list)))
                # add y
                phonem_idx = TimitDataset.phonemes.index(ph)
                y_list.append(phonem_idx)

            # early break for debugging
            it += 1
            #if it == 20:
            #    break

        # return np arrays, second dimension can vary
        x = np.array(x_list)
        y = np.array(y_list)
        return x, y


    def load_timit_data(self, root_dir):

        x_list = []
        y_list = []
        wav_glob = os.path.join(root_dir, '**/*.wav')
        it = 1
        for wav_filename in glob.glob(wav_glob, recursive=True):
            if wav_filename[-7:] in ['SA1.wav', 'SA2.wav']:
                continue
            # load audio and get mfcc, add x
            sample_rate, wav = sciwav.read(wav_filename)
            mfcc_data = psf.mfcc(wav, samplerate=sample_rate, numcep=self.mfcc_size)
            x_list.append(mfcc_data)
            if len(x_list) % 100 == 0:
                print('Loaded {} files.'.format(len(x_list)))
            
            # parse the text file with phonemes, and add y
            phn_filename = wav_filename[:-3] + 'PHN' # fragile, i know
            with open(phn_filename) as f:
                phonemes = [line.split()[2] for line in f.readlines()]
                phonem_idxs = np.array([TimitDataset.phonemes.index(ph) for ph in phonemes])
                y_list.append(phonem_idxs)

            # early break for debugging
            it += 1
            #if it == 200:
            #    break

        # return np arrays, second dimension can vary
        x = np.array(x_list)
        y = np.array(y_list)
        return x, y