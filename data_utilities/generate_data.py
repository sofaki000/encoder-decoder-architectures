import numpy as np
import random


def generate_single_seq(length=30, min_len=5, max_len=10):
    # https://medium.com/@devnag/pointer-networks-in-tensorflow-with-sample-code-14645063f264
    """ Generates a sequence of numbers of random length and inserts a sub-sequence oh greater numbers at random place
    Input:
    length: total sequence length
    min_len: minimum length of sequence
    max_len: maximum length of sequence
    Output:
    sequence of numbers, index of the start of greater numbers subsequence"""
    seq_before = [(random.randint(1, 5)) for x in range(random.randint(min_len, max_len))]
    seq_during = [(random.randint(6, 10)) for x in range(random.randint(min_len, max_len))]
    seq_after = [random.randint(1, 5) for x in range(random.randint(min_len, max_len))]
    seq = seq_before + seq_during + seq_after
    seq = seq + ([0] * (length - len(seq)))
    return (seq, len(seq_before), len(seq_before) + len(seq_during)-1)

def train_test_split(data, labels, split=0.8):
    '''

    split time series into train/test sets

    : param t:                      time array
    : para y:                       feature array
    : para split:                   percent of data to include in training set
    : return t_train, y_train:      time/feature training and test sets;
    :        t_test, y_test:        (shape: [# samples, 1])

    '''
    data_len = len(data)
    indx_split = int(split * data_len)

    x_train = data[:indx_split]
    y_train = labels[:indx_split]

    x_test = data[indx_split:]
    y_test = labels[indx_split:]

    return x_train, y_train, x_test, y_test


def task_data_for_predicting_same_sequence(n_samples, seq_len):
    # Boundary tasks
    data, labels = [], []
    for _ in range(n_samples):
        input = np.random.permutation(range(seq_len)).tolist()
        target = input
        data.append(input)
        labels.append(target)
    return data, labels

def generate_set_seq(N):
    # generate Boundary tasks
    """Generates a set of N sequences of fixed length"""
    data = []
    starts = []
    ends = []
    for i in range(N):
        seq, ind_start, ind_end = generate_single_seq()
        data.append(seq)
        starts.append(ind_start)
        ends.append(ind_end)
    return data, starts, ends


def make_seq_data(n_samples, seq_len):
    # Boundary tasks
    data, labels = [], []
    for _ in range(n_samples):
        input = np.random.permutation(range(seq_len)).tolist()
        target = sorted(range(len(input)), key=lambda k: input[k])
        data.append(input)
        labels.append(target)
    return data, labels