import argparse
import copy
import json

import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd

# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    n_classes = len(P)

    m = y.shape[0]
    print(m)
    new_y = y.copy()

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        new_y[idx] = np.random.choice(n_classes, 1, p=P[i, :])[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    print(P)

    return y_train, actual_noise


def noisify(nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--label_vocab_path', type=str)
parser.add_argument('--out_path', type=str)
parser.add_argument('--noise_type', type=str)
parser.add_argument('--noise_rate', type=float)
args = parser.parse_args()

random_state = 2020


if __name__ == '__main__':
    data = pd.read_csv(args.data_path)
    with open(args.label_vocab_path) as f:
        label_vocab = json.load(f)
    id2label = {val: key for key, val in label_vocab.items()}

    mask = data['数据集'] == '训练'
    train_data = data[mask].copy(deep=True)
    test_data = data[~mask]

    y_train = train_data['label'].apply(lambda x: label_vocab[x]).values
    y_train, actual_noise_rate = noisify(len(label_vocab), y_train, args.noise_type, args.noise_rate, random_state)
    new_label = [id2label[i] for i in y_train]
    # print(type(new_label[0]), type(train_data['label'].iloc[0]))
    train_data['clean'] = [1 if new_label[i] == train_data['label'].iloc[i] else 0 for i in range(len(new_label))]
    train_data['label'] = new_label

    ret = pd.concat([train_data, test_data])
    ret.sort_values(by=['label', '数据集'], inplace=True)
    ret.to_csv(args.out_path, index=False)
