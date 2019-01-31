from sklearn import svm
from data_loader import get_dataset

train_x, train_y, test_x, test_y = get_dataset()  # use pca

for penalty in [1, 10, 100]:
    print('C = %d' % penalty)
    for kernel in ['linear', 'rbf']:
        print(kernel)
        clf = svm.SVC(kernel=kernel, C=penalty)
        clf.fit(train_x, train_y)
        print('score: %.4f' % clf.score(test_x, test_y))
        print('support vector: %d' % clf.support_vectors_.shape[0])
