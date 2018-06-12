import numpy as np

def dense_to_sparse(dense, dtype=np.int32):
    indices = []
    values = []
    # dense: [batch_sz x time?], phonems

    for n, times in enumerate(dense):
        indices.extend(zip([n]*len(times), range(len(times))))
        values.extend(times)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(dense), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def pad_time(dense):
    # dense: [batch_sz x time? x mfcc_len]
    # treats it as [batch_sz x variable_len? x [...data...]]
    batch_size = len(dense)
    time_lens = np.asarray([len(times) for times in dense], dtype=np.int64)
    max_time = np.max(time_lens)

    # take the constant shape of inner data from first sample
    data_shape = np.asarray(dense[0]).shape[1:]

    # create an empty box with zeros to place parts of 'dense'
    padded = (np.ones((batch_size, max_time) + data_shape) * 0.0).astype(np.float32)

    # fill in rows of the box with time x data
    for idx, times in enumerate(dense):
        assert(times.shape[1:] == data_shape)
        padded[idx, :len(times)] = times

    # return padded and the lengths array
    return padded, time_lens
