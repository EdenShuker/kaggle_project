from csv import DictReader

import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import ExeDataset


class MalConv(nn.Module):
    def __init__(self, labels, input_length=2000000, window_size=500):
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 9)

        self.sigmoid = nn.Sigmoid()

        self.i2l = {i: l for i, l in enumerate(labels)}

    def forward(self, x):
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

        return nn.Softmax(x)


def split_csv_dict(csv_filepath):
    fps = []
    labels = []

    for row in DictReader(csv_filepath):
        fps.append(row['Id'])
        labels.append(row['Class'])

    return fps, labels


def train_on(first_n_byte=2000000):
    model = MalConv()

    fps_train, y_train = split_csv_dict('train_set.csv')
    fps_dev, y_dev = split_csv_dict('test_set.csv')

    files_dirpath = '../data/files'
    dataloader = DataLoader(ExeDataset(fps_train, files_dirpath, y_train, first_n_byte),
                            batch_size=1, shuffle=True, num_workers=1)
    validloader = DataLoader(ExeDataset(fps_dev, files_dirpath, y_dev, first_n_byte),
                             batch_size=1, shuffle=False, num_workers=1)

    bce_loss = nn.BCELoss()
    lr = 0.001
    adam_optim = torch.optim.Adam(model.parameters(), lr)

    valid_best_acc = 0.0
    total_step = 0

    max_step = 1
    test_step = 10

    while total_step < max_step:

        # Training
        for batch_data in dataloader:
            start = time.time()

            adam_optim.zero_grad()

            exe_input = batch_data[0]
            exe_input = Variable(exe_input.long(), requires_grad=False)

            label = batch_data[1]
            label = Variable(label.float(), requires_grad=False)

            pred = model(exe_input)
            loss = bce_loss(pred, label)
            loss.backward()
            adam_optim.step()

            step_cost_time = time.time() - start

            total_step += 1

            # Interupt for validation
            if total_step % test_step == 0:
                curr_acc = validate_dev_set(validloader, model)
                print 'time to train:', step_cost_time, 'current-accuracy:', curr_acc
                if curr_acc > valid_best_acc:
                    valid_best_acc = curr_acc
                    torch.save(model, 'model.file')


def validate_dev_set(validloader, model):
    good = 0.0
    for val_batch_data in validloader:

        exe_input = val_batch_data[0]
        exe_input = Variable(exe_input.long(), requires_grad=False)

        labels = val_batch_data[1]
        labels = Variable(labels.float(), requires_grad=False)

        preds = model(exe_input).npvalue()
        for vec, label in preds, labels:
            pred_label = model.i2l(np.argmax(vec))
            if pred_label == label:
                good += 1
    return good / len(validloader)
