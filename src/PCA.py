import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca_exp():
    data = np.load('../data/data.npy')

    sum_variance = []
    for dim in range(1, 1500, 50):
        print(dim)
        pca = PCA(n_components=dim)
        pca.fit(data)
        sum_variance.append(sum(pca.explained_variance_ratio_))

    plt.plot([range(1, 1500, 50)], sum_variance)
    plt.xlabel('dim')
    plt.ylabel('total variance ratio')
    plt.xlim([0, 1500])
    plt.ylim([0, 1])
    plt.grid(True)
    plt.show()


def pca_data():
    train_x = np.load('../data/train_x.npy')
    test_x = np.load('../data/test_x.npy')
    pca = PCA(n_components=0.95)

    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    print(pca.n_components_)
    print(train_x.shape)
    print(test_x.shape)

    np.save('../data/train_x_pca.npy', train_x)
    np.save('../data/test_x_pca.npy', test_x)


if __name__ == '__main__':
    pca_data()
