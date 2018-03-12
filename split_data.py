from collections import OrderedDict
from operator import itemgetter

import utils
import random


def train_test_split(path_to_files, test_size=0.33):
    files_to_labels = utils.get_f2l_dict(path_to_files)
    labels_to_files = get_labels_to_files(files_to_labels)
    train_set = {}
    test_set = {}
    for label in labels_to_files:
        files_with_label = labels_to_files[label]
        num_samples = len(files_with_label)

        if num_samples == 1:  # only one sample of that label, add it to train
            train_set[files_with_label[0]] = label
            continue

        # separate the indexes into train and test
        num_test_samples = int(num_samples * test_size)
        if num_test_samples == 0:
            num_test_samples = 1
        samples_indexes = range(0, num_samples)
        random.shuffle(samples_indexes)
        test_samples_indexes = samples_indexes[:num_test_samples]
        train_samples_indexes = samples_indexes[num_test_samples:]

        # add to each list the needed samples from X and the connected label
        for test_sample_i in test_samples_indexes:
            test_set[files_with_label[test_sample_i]] = label
        for train_sample_i in train_samples_indexes:
            train_set[files_with_label[train_sample_i]] = label

    save_to_csv_file(train_set, 'train_set')
    save_to_csv_file(test_set, 'test_set')


def save_to_csv_file(files_set, f_name, dirpath='data'):
    csv_f = open(dirpath + '/' + f_name + '.csv', 'w')
    csv_f.write('Id,Class\n')
    files_set = OrderedDict(sorted(files_set.items(), key=itemgetter(1)))
    for f in files_set.iterkeys():
        csv_f.write('%s,%s\n' % (f, files_set[f]))  # write the file and its label
    csv_f.close()


def get_labels_to_files(files_to_labels):
    labels_to_files = dict()
    for f in files_to_labels:
        label = files_to_labels[f]
        if not labels_to_files.has_key(label):
            labels_to_files[label] = [f]
        else:
            labels_to_files[label].append(f)
    return labels_to_files


if __name__ == '__main__':
    train_test_split('data/train_labels_filtered.csv')
