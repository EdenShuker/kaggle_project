from csv import DictReader

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import ExeDataset

class MalConv(nn.Module):
    def __init__(self, input_length=2000000, window_size=500):
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 9)

        self.sigmoid = nn.Sigmoid()

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


first_n_byte = 2000000
model = MalConv()

csv_dict = DictReader(open('train_set'))



dataloader = DataLoader(ExeDataset(list(tr_table.index), '../data/files', list(tr_table.ground_truth),first_n_byte),
                            batch_size=1, shuffle=True, num_workers=1)
validloader = DataLoader(ExeDataset(list(val_table.index), valid_data_path, list(val_table.ground_truth),first_n_byte),
                        batch_size=1, shuffle=False, num_workers=1)
