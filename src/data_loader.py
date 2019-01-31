import numpy as np


def get_dataset(use_pca=True):
    if use_pca:
        train_x = np.load('../data/train_x_pca.npy')
        test_x = np.load('../data/test_x_pca.npy')
    else:
        train_x = np.load('../data/train_x.npy')
        test_x = np.load('../data/test_x.npy')

    train_y = np.load('../data/train_y.npy')
    test_y = np.load('../data/test_y.npy')

    return train_x, train_y, test_x, test_y
