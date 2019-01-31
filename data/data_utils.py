import numpy as np

lab_idx = 7     # Characteristics [DiseaseState]
min_sample_num = 20
sample_num = 5896
gene_num = 22283

data_in = './Gene_Chip_Data/microarray.original.txt'
label_in = './Gene_Chip_Data/E-TABM-185.sdrf.txt'


def data_process():
    # read label
    with open(label_in, 'r') as fl:
        line = fl.readline()
        print('select label: ', line.split('\t')[lab_idx])

        label = []
        label_dist = {}     # label: num

        for i in range(sample_num):
            sample = fl.readline()
            lab = sample.split('\t')[lab_idx]
            label.append(lab)
            if lab.strip():
                if lab in label_dist:
                    label_dist[lab] += 1
                else:
                    label_dist[lab] = 1

    # select label
    selected_label = []
    for lab, num in label_dist.items():
        if num > min_sample_num:
            selected_label.append(lab)
    print('select %d types of disease' % len(selected_label))

    # load data
    data = []
    with open(data_in, 'r') as fd:
        fd.readline()
        for i in range(gene_num):
            line = fd.readline().split('\t')[1:]
            line = [float(x) for x in line]
            data.append(line)
    data = np.array(data).T

    # select data
    label_idx = []
    remind_data_idx = []
    for i in range(sample_num):
        lab = label[i]
        if lab in selected_label:
            label_idx.append(selected_label.index(lab))
            remind_data_idx.append(i)
    data = data[remind_data_idx, :]
    print('select %d samples' % data.shape[0])

    # write into files
    np.save('data.npy', data)
    np.save('label_idx.npy', label_idx)

    with open('label.txt', 'w') as f:
        for lab in selected_label:
            f.write(lab)
            f.write('\n')


def split_train_test(data_file, test_ratio):
    label_idx = np.load('label_idx.npy')
    data = np.load(data_file)

    n_samples = label_idx.size
    n_test = int(n_samples * test_ratio)
    order = np.random.permutation(n_samples)
    train_order = order[n_test:]
    test_order = order[:n_test]
    print('train set: %d' % (n_samples - n_test))
    print('test set: %d' % n_test)

    np.save('train_x.npy', data[train_order, :])
    np.save('train_y.npy', label_idx[train_order])
    np.save('test_x.npy', data[test_order, :])
    np.save('test_y.npy', label_idx[test_order])


if __name__ == '__main__':
    # data_process()
    split_train_test('data.npy', 0.1)
