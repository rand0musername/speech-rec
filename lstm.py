import numpy as np
import tensorflow as tf
import random
import math
from timit_dataset import TimitDataset
from conversions import *

class DeepBiLstmCtc():

    def __init__(self, 
                 mfcc_size, 
                 lstm_num_hidden,
                 num_layers,
                 num_classes,
                ):

        # inputs
        # MFCC vectors
        # max_num_windows_for_batch = max_time
        # size [batch_size, max_time, mfcc_size]
        self.input_x = tf.placeholder(tf.float32, [None, None, mfcc_size], name='input_x')
        # ctc_loss needs sparse tensor and an array of num_windows
        # indices[i, :] = [batch, time], values[i] is in [0, num_labels)
        self.input_y = tf.sparse_placeholder(tf.int32, name='input_y')
        # [batch_size], holds max_time values
        self.len_seq = tf.placeholder(tf.int32, [None], name='len_seq') 

        # architecture, deep bidirectional lstm
        cells_fwd = []
        cells_bwd = []
        for idx in range(num_layers):
            cells_fwd.append(tf.nn.rnn_cell.LSTMCell(lstm_num_hidden, state_is_tuple=True))
            cells_bwd.append(tf.nn.rnn_cell.LSTMCell(lstm_num_hidden, state_is_tuple=True))
        layers_fwd = tf.nn.rnn_cell.MultiRNNCell(cells_fwd, state_is_tuple=True)
        layers_bwd = tf.nn.rnn_cell.MultiRNNCell(cells_bwd, state_is_tuple=True)
        self.out, _ = tf.nn.bidirectional_dynamic_rnn(layers_fwd, layers_bwd, self.input_x, self.len_seq, dtype=tf.float32)
        self.full_out = tf.concat(self.out, 2) 
        # [batch_size, max_time, 2xlstm_num_hidden]

        # dense layer to get num_classes
        self.out_flat = tf.reshape(self.full_out, [-1, 2*lstm_num_hidden])
        self.logits = tf.layers.dense(self.out_flat, num_classes)
        # [batch_size, max_time, num_classes]

        # reshape back and do time major to prepare for ctc_loss
        batch_sz = tf.shape(self.input_x)[0]
        self.logits = tf.reshape(self.logits, [batch_sz, -1, num_classes])
        self.logits = tf.transpose(self.logits, (1, 0, 2))  
        # [max_time, batch_size, num_classes]

        # get ctc loss and cost
        loss = tf.nn.ctc_loss(self.input_y, self.logits, self.len_seq)
        self.total_loss = tf.reduce_mean(loss)

        # set up momentum optimizer
        initial_learning_rate = 1e-3
        momentum = 0.9
        self.optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           momentum).minimize(self.total_loss)

        # decode and calculate edit distance and ler inaccuracy
        decoded, _ = tf.nn.ctc_greedy_decoder(self.logits, self.len_seq)
        self.decoded = tf.cast(decoded[0], tf.int32)
        # sparse tensor: [batch_size, time] -> in range [0, nc-1]
        # input_y is also sparse: [batch_size, time] -> in range [0, nl-1]
        # num classes is 62, num labels is 61
        # inaccuracy: label error rate
        self.ler = tf.reduce_mean(tf.edit_distance(self.decoded, self.input_y)) # [0, 1]

        # summarize metrics for tensorboard
        loss_summary = tf.summary.scalar("batch_loss", self.total_loss)
        ler_summary = tf.summary.scalar("batch_ler", self.ler)
        self.summary_op = tf.summary.merge_all()

        # prepare tensorboard writer
        self.writer = tf.summary.FileWriter('./logs/train_real_test')
        self.writer.add_graph(tf.get_default_graph())

        # save the model after training
        self.saver = tf.train.Saver()

    def train_and_test(self, dataset, batch_size, num_epochs):
        self.batch_size = batch_size

        with tf.Session() as sess:
            # init global vars
            sess.run(tf.global_variables_initializer())

            # get training data
            num_examples = dataset.get_num_training_examples()
            num_batches = math.ceil(num_examples / batch_size)

            # train several epochs
            for epoch_idx in range(num_epochs):
                print("Epoch " + str(epoch_idx) + " starting.")
                epoch_loss = 0
                epoch_ler = 0

                # train several batches
                for batch_idx in range(num_batches):
                    # both num_windows and num_phonemes are time
                    # [batch_size num_windows? mfcc_size] [batch_size num_phonemes?]
                    x_batch, y_batch = dataset.get_training_batch(batch_idx, batch_size)
                    actual_batch_size = x_batch.shape[0]

                    # pad second dim of x_batch
                    x_batch_padded, x_batch_len_seq = pad_time(x_batch) 

                    # convert y_batch to a sparse vector for ctc
                    y_batch_sparse = dense_to_sparse(y_batch)

                    # evaluate and get loss
                    results = [self.ler, 
                               self.total_loss, 
                               self.optimizer, 
                               self.summary_op]
                    feed_dict = {self.input_x: x_batch_padded, 
                                 self.input_y: y_batch_sparse,
                                 self.len_seq: x_batch_len_seq}
                    batch_ler, batch_loss, _, summary = sess.run(results, feed_dict)

                    # add loss/ler to epoch metrics and output to tensorboard
                    self.writer.add_summary(summary, epoch_idx * num_batches + batch_idx)
                    epoch_loss += batch_loss*actual_batch_size
                    epoch_ler += batch_ler*actual_batch_size

                    # print batch loss
                    print("Epoch:", '{0:3d}'.format(epoch_idx), 
                          "|Batch:", '{0:3d}'.format(batch_idx), 
                          "|BatchLoss:", '{0:8.4f}'.format(batch_loss),
                          "|BatchLer:", '{0:8.4f}'.format(batch_ler))

                # average per example
                epoch_loss /= num_examples
                epoch_ler /= num_examples
                print("Epoch over:", '{0:3d}'.format(epoch_idx))
                print("MeanEpochLoss:", '{0:3f}'.format(epoch_loss))
                print("MeanEpochLer:", '{0:3f}'.format(epoch_ler))

            # save and test
            self.saver.save(sess, './saved_model/dblc-3-150')
            self.test_on_random_training_batch(sess, dataset, batch_size)
            self.test(sess, dataset)

    def test(self, sess, dataset):
        print('===Testing===')
        # calculate metrics on the test set
        num_examples = dataset.get_num_test_examples()
        num_batches = math.ceil(num_examples / self.batch_size)
        all_pairs = []
        test_loss = 0
        test_ler = 0
        for batch_idx in range(num_batches):
                x_batch, y_batch = dataset.get_test_batch(batch_idx, self.batch_size)
                actual_batch_size = x_batch.shape[0]
                loss, ler, pairs = self.evaluate_and_decode(sess, x_batch, y_batch)
                all_pairs.extend(pairs)
                test_loss += loss*actual_batch_size
                test_ler += ler*actual_batch_size
        test_loss /= num_examples
        test_ler /= num_examples
        self.log(test_loss, test_ler, random.sample(all_pairs, min(10, len(pairs))))

    def test_on_random_training_batch(self, sess, dataset, batch_size):
        print('===Random training batch===')
        # calculate metrics on a random training batch
        num_examples = dataset.get_num_training_examples()
        num_batches = math.ceil(num_examples / batch_size)
        idx = random.randint(0, num_batches)
        x_batch, y_batch = dataset.get_training_batch(idx, batch_size)

        loss, ler, pairs = self.evaluate_and_decode(sess, x_batch, y_batch)
        self.log(loss, ler, random.sample(pairs, min(10, len(pairs))))

    def log(self, loss, ler, pairs):
        # log metrics
        print("Loss:", '{0:3f}'.format(loss))
        print("Ler:", '{0:3f}'.format(ler))
        for pair in pairs:
            print('\t Target: %s' % pair[0])
            print('\t Decoded: %s' % pair[1])

    def evaluate_and_decode(self, sess, xs, ys):
        # evaluate xs and compare to ys
        xs_padded, xs_len_seq = pad_time(xs) 
        ys_sparse = dense_to_sparse(ys)

        results = [self.total_loss, self.ler, self.decoded]
        feed_dict = {self.input_x: xs_padded, 
                     self.input_y: ys_sparse,
                     self.len_seq: xs_len_seq}
        loss, ler, d = sess.run(results, feed_dict)

        dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)

        # convert to strings and get 10 pairs of sequences to return
        it = 0
        pairs = []
        for batch_idx, seq in enumerate(dense_decoded):
            it += 1
            seq = [TimitDataset.phonemes[s] for s in seq if s != -1]
            target_seq = [TimitDataset.phonemes[s] for s in ys[batch_idx]]
            pairs.append((' '.join(target_seq), ' '.join(seq)))
            if it == 10:
                break
        return loss, ler, pairs