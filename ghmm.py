import numpy as np
import random
import math
from timit_dataset import TimitDataset
from conversions import *
from hmmlearn import hmm

class Ghmm():
    """ A gaussian HMM for phonem sequence classification """

    def __init__(self, mfcc_size, n_states, n_iter, num_labels):
        self.n_states = n_states  # the number of HMM states 
        self.mfcc_size = mfcc_size   # the size of a single MFCC vector
        self.hmms = []    # one hmm for each phonem
        self.phonem_freq = {}    # language model: phoneme frequencies
        self.num_labels = num_labels    # number of phonemes
        # create an hmm 
        for idx in range(num_labels):
            self.hmms.append(hmm.GaussianHMM(n_components=n_states, n_iter=n_iter))

    def train_and_test(self, dataset, batch_size):
        self.batch_size = batch_size
        x_train, y_train = dataset.get_training_data()
        num_examples = dataset.get_num_training_examples()
        dropped = 0
        for idx in range(num_examples):
            if idx % 100 == 0:
                print('Training on {} out of {} examples'.format(idx, num_examples))
            mfcc_vec_seq = x_train[idx]
            phoneme_idx = y_train[idx]
            # drop sequences that are too short
            if len(mfcc_vec_seq) < self.n_states:
                print('dropped')
                dropped += 1
            else:
                # find an appropriate hmm and fit using EM
                hmm = self.hmms[phoneme_idx]
                # add phonem frequency
                if not phoneme_idx in self.phonem_freq:
                    self.phonem_freq[phoneme_idx] = 0
                self.phonem_freq[phoneme_idx] += 1
                hmm.fit(mfcc_vec_seq)
        # calculate phonem frequences
        phonem_num = sum(self.phonem_freq.values())
        for key in self.phonem_freq:
            self.phonem_freq[key] /= float(phonem_num)
            print(self.phonem_freq[key])
            print(key)
        print('Done training, dropped {} out of {}'.format(dropped, num_examples))
        self.test_on_random_training_batch(dataset, self.batch_size)
        self.test(dataset)

    def test(self, dataset):
        print("==Testing==")
        num_examples = dataset.get_num_test_examples()
        num_batches = math.ceil(num_examples / self.batch_size)
        all_pairs = []
        test_ler = 0
        for batch_idx in range(num_batches):
                if batch_idx % 10 == 0:
                    print('Testing {}'.format(batch_idx))
                x_batch, y_batch = dataset.get_test_batch(batch_idx, self.batch_size)
                actual_batch_size = x_batch.shape[0]
                ler, pairs = self.evaluate_batch(x_batch, y_batch)
                test_ler += ler * actual_batch_size
                all_pairs.extend(pairs)
        test_ler /= num_examples
        self.log(test_ler, random.sample(all_pairs, 10))


    def test_on_random_training_batch(self, dataset, batch_size):
        print('===Random training batch===')
        num_examples = dataset.get_num_training_examples()
        num_batches = math.ceil(num_examples / batch_size)
        idx = random.randint(0, num_batches)
        x_batch, y_batch = dataset.get_training_batch(idx, batch_size)
        ler, pairs = self.evaluate_batch(x_batch, y_batch)
        self.log(ler, random.sample(pairs, 10))

    def evaluate_batch(self, xs, ys):
        fails = 0
        actual_batch_size = xs.shape[0]
        pairs = []
        # calculate error rate on a batch
        for idx in range(actual_batch_size):
            mfcc_vec_seq = xs[idx]
            target_phoneme = ys[idx]
            guessed_phoneme = self.infer(mfcc_vec_seq)
            if guessed_phoneme != target_phoneme:
                fails += 1
            pairs.append((TimitDataset.phonemes[target_phoneme], 
                          TimitDataset.phonemes[guessed_phoneme]))
        ler = float(fails) / actual_batch_size
        return ler, pairs


    def log(self, ler, pairs):
        print("Ler:", '{0:3f}'.format(ler))
        for pair in pairs:
            print('\t Target: <%s>' % pair[0])
            print('\t Decoded: <%s>' % pair[1])

    def infer(self, mfcc_vec_seq):
        # infer the most likely phoneme using the forward algorithm and bayes
        scores = []
        for idx in range(self.num_labels):
            if idx not in self.phonem_freq:
                continue  # not fitted at all, no examples of this phoneme were seen
            score = self.phonem_freq[idx] * math.exp(self.hmms[idx].score(mfcc_vec_seq))
            scores.append((score, idx))
        best_idx = max(scores)[1]
        return best_idx