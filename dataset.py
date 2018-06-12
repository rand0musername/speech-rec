class Dataset():
    """ A generic dataset class offering common accessors """
    def __init__(self):
        pass

    def get_num_training_examples(self):
        return self.x_train.shape[0]

    def get_training_batch(self, batch_idx, batch_size):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size # if there is less that's ok

        x_batch = self.x_train[start_idx:end_idx]
        y_batch = self.y_train[start_idx:end_idx]

        return x_batch, y_batch

    def get_num_test_examples(self):
        return self.x_test.shape[0]

    def get_test_batch(self, batch_idx, batch_size):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size # if there is less that's ok

        x_batch = self.x_test[start_idx:end_idx]
        y_batch = self.y_test[start_idx:end_idx]

        return x_batch, y_batch

    def get_test_data(self):
        return self.x_test, self.y_test

    def get_training_data(self):
        return self.x_train, self.y_train