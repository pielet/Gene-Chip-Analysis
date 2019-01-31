import os
import time
import numpy as np
import tensorflow as tf
from model import GeneModel

from data_loader import get_dataset

learning_rate = 1e-3
batch_size = 64
epochs = 50
use_res = True
use_dropout = True
use_norm = False

load_model = False

ckpt_path = 'res_dropout/ckpt'
store_path = 'res_dropout'


def create_model(sess):
    model = GeneModel(learning_rate, use_res, use_dropout, use_norm)

    if not load_model:
        print("Creating model with fresh parameters")
        sess.run(tf.global_variables_initializer())
        return model

    # load a previously saved model
    ckpt = tf.train.latest_checkpoint(ckpt_path)
    if ckpt:
        print('loading model {}'.format(os.path.basename(ckpt)))
        model.saver.restore(sess, ckpt)
        return model
    else:
        raise (ValueError, 'can NOT find model')


def evaluate_batch(logit, gt, n_samples=batch_size):
    max_logit = np.max(logit, axis=1)
    classes = np.zeros([n_samples])

    for i in range(n_samples):
        classes[i] = np.where(logit[i] == max_logit[i])[0][0]

    gt_classes = np.where(gt == 1)[1]

    return np.sum(gt_classes == classes)


def get_batches(x, y, y_test):
    n_train = y.size
    n_test = y_test.size

    # one-hot
    new_y = np.zeros([n_train, 41])
    new_y_test = np.zeros([n_test, 41])
    for i, idx in enumerate(y):
        new_y[i][idx] = 1
    for i, idx in enumerate(y_test):
        new_y_test[i][idx] = 1

    n_train_batches = n_train // batch_size
    n_train_extra = n_train - n_train_batches*batch_size
    x = np.split(x[:-n_train_extra, :], n_train_batches)
    y = np.split(new_y[:-n_train_extra, :], n_train_batches)

    return x, y, new_y_test


def train():
    print('loading data')
    x, y, x_test, y_test = get_dataset()    # use pca

    n_train = x.shape[0]
    n_test = x_test.shape[0]
    n_batches = n_train // batch_size
    print('train samples: %d' % n_train)
    print('test samples: %d' % n_test)

    x, y, y_test = get_batches(x, y, y_test)

    config = tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        print('creating model')
        model = create_model(sess)

        start_time = time.time()
        total_loss = []
        train_acc = []
        test_acc = []
        # GO !!
        for e in range(epochs):
            print('working on epoch {0}/{1}'.format(e + 1, epochs))
            epoch_start_time = time.time()
            epoch_loss, epoch_acc = 0, 0

            for i in range(n_batches):
                print('working on epoch {0}, batch {1}/{2}'.format(e+1, i+1, n_batches))
                enc_in, dec_out = x[i], y[i]
                _, output, step_loss, _ = model.step(sess, enc_in, dec_out)
                step_acc = evaluate_batch(output, dec_out) / batch_size
                epoch_loss += step_loss
                epoch_acc += step_acc
                print('current batch loss: {:.2f}'.format(step_loss))

            epoch_time = time.time() - epoch_start_time
            print('epoch {0}/{1} finish in {2:.2f} s'.format(e+1, epochs, epoch_time))
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            total_loss.append(epoch_loss)
            train_acc.append(epoch_acc)
            print('average epoch loss: {:.4f}'.format(epoch_loss))
            print('average epoch acc: {:.4f}'.format(epoch_acc))

            print('saving model...')
            model.saver.save(sess, ckpt_path, model.global_step.eval())

            # test after each epoch
            output = model.step(sess, x_test, y_test, is_train=False)[0]
            step_acc = evaluate_batch(output, y_test, n_test) / n_test
            test_acc.append(step_acc)
            print('test acc: %.4f\n' % step_acc)

        print('training finish in {:.2f} s'.format(time.time() - start_time))

        with open(os.path.join(store_path, 'summary.txt'), 'w') as f:
            for i in range(epochs):
                f.write('{0}\t{1}\t{2}\n'.format(total_loss[i], train_acc[i], test_acc[i]))


if __name__ == '__main__':
    train()
