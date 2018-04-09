from csv import DictReader
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import ExeDataset


class MalConv(nn.Module):
    def __init__(self, labels, input_length=2000000, window_size=500):
        """
        :param labels: list of optional labels.
        :param input_length: length of input.
        :param window_size: size of window
        """
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 9)

        self.sigmoid = nn.Sigmoid()

        self.i2l = {i: l for i, l in enumerate(labels)}
        self.l2i = {l: i for i, l in self.i2l.iteritems()}

    def forward(self, x):
        """
        :param x: Variable input.
        :return: Variable output.
        """
        x = self.embed(x)
        # Channel first
        x = torch.transpose(x, -1, -2)

        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))

        x = cnn_value * gating_weight
        x = self.pooling(x)

        x = x.view(-1, 128)
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x


def split_csv_dict(csv_filepath):
    """
    :param csv_filepath: string representing file-path to csv file,
                         must contain the following columns: Id, Class
    :return: two lists, first is of the Id-column and second of Class-column.
    """
    fps = []
    labels = []

    for row in DictReader(open(csv_filepath)):
        fps.append(row['Id'])
        labels.append(row['Class'])

    return fps, labels


def train_on(first_n_byte=2000000, lr=0.001, verbose=True, num_epochs=10):
    """
    :param first_n_byte: number of bytes to read from each file.
    :param lr: learning rate.
    :param verbose: boolean.
    :param num_epochs: number of epochs.
    """
    # create model
    model = MalConv(range(1, 10))
    l2i = model.l2i

    # load data
    fps_train, y_train = split_csv_dict('train_set.csv')
    fps_dev, y_dev = split_csv_dict('test_set.csv')

    # transfer data to DataLoader object
    files_dirpath = '../data/files/'
    dataloader = DataLoader(ExeDataset(fps_train, files_dirpath, y_train, l2i, first_n_byte),
                            batch_size=1, shuffle=True, num_workers=1)
    validloader = DataLoader(ExeDataset(fps_dev, files_dirpath, y_dev, l2i, first_n_byte),
                             batch_size=1, shuffle=False, num_workers=1)

    cross_entropy_loss = nn.CrossEntropyLoss()
    adam_optim = torch.optim.Adam(model.parameters(), lr)

    valid_best_acc = 0.0
    total_step = 0
    test_step = 20

    for epoch in range(num_epochs):
        t0 = time()
        good = 0.0

        for batch_data in dataloader:
            adam_optim.zero_grad()

            exe_input, label = batch_data[0], batch_data[1]
            exe_input, label = Variable(exe_input.long()), Variable(label.long()).squeeze()
            pred = model(exe_input)

            gold_label = label.data.numpy()[0]
            pred_label = torch.max(pred, 1)[1].data.numpy()[0]
            gold_label, pred_label = model.i2l[gold_label], model.i2l[pred_label]

            if verbose:
                print 'gold: ', gold_label, ', pred: ', pred_label
            if gold_label == pred_label:
                good += 1

            loss = cross_entropy_loss(pred, label)
            loss.backward()
            adam_optim.step()

            total_step += 1

            if total_step % test_step == test_step - 1:  # interrupt for validation
                curr_acc = validate_dev_set(validloader, model, verbose)
                if curr_acc > valid_best_acc:  # update best accuracy
                    valid_best_acc = curr_acc
                    torch.save(model, 'model.file')
        acc = good / len(y_train)
        print str(epoch) + 'TRN\ttime:', time() - t0, ', accuracy:', acc * 100, '%'


def validate_dev_set(valid_loader, model, verbose=True):
    """
    check performance of model on dev-set.
    :param valid_loader: DataLoader.
    :param model: Module which is the model to check with.
    :param verbose: boolean.
    :return: model accuracy on dev-set.
    """
    print '##########\tDEV\t##########'
    t0 = time()
    good = 0.0

    for val_batch_data in valid_loader:
        exe_input, labels = val_batch_data[0], val_batch_data[1]
        exe_input, labels = Variable(exe_input.long(), requires_grad=False), \
                            Variable(labels.long(), requires_grad=False)
        preds, labels = model(exe_input).data.numpy(), labels.data.numpy()

        for pred, gold_label in zip(preds, labels):
            pred_label, gold_label = np.argmax(pred), gold_label[0]
            pred_label, gold_label = model.i2l[pred_label], model.i2l[gold_label]

            if verbose:
                print 'gold: ', gold_label, ', pred: ', pred_label

            if gold_label == pred_label:
                good += 1
    acc = good / len(valid_loader)
    print ' DEV\ttime:', time() - t0, ', accuracy:', acc * 100, '%\n'
    return acc


if __name__ == '__main__':
    train_on(verbose=True)
