import numpy as np
from sklearn.linear_model import LogisticRegression
from data_loader import get_dataset

C = [10, 100, 1000]
train_x, train_y, test_x, test_y = get_dataset()    # use pca

for c in C:
    LR_l1 = LogisticRegression(penalty='l1', C=c)
    LR_l2 = LogisticRegression(penalty='l2', C=c)

    LR_l1.fit(train_x, train_y)
    LR_l2.fit(train_x, train_y)

    score_l1 = LR_l1.score(test_x, test_y)
    score_l2 = LR_l2.score(test_x, test_y)

    coef_l1 = LR_l1.coef_.ravel()
    coef_l2 = LR_l2.coef_.ravel()

    sparsity_l1 = np.mean(coef_l1 == 0) * 100
    sparsity_l2 = np.mean(coef_l2 == 0) * 100

    print('C = %d' % c)
    print("Sparsity with L1 penalty: %.2f%%" % sparsity_l1)
    print("score with L1 penalty: %.4f" % score_l1)
    print("Sparsity with L2 penalty: %.2f%%" % sparsity_l2)
    print("score with L2 penalty: %.4f" % score_l2)