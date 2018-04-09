import numpy as np
from torch.utils.data import Dataset


class ExeDataset(Dataset):
    def __init__(self, fp_list, data_path, label_list, l2i, first_n_byte=2000000):
        """
        :param fp_list: list of strings, each is file-path.
        :param data_path: string which is a path to the directory where the files mentioned in fp_list are in.
        :param label_list: list og labels, each is integer.
        :param l2i: dict that maps label to index.
        :param first_n_byte: number of bytes to read from each file.
        """
        self.fp_list = fp_list
        self.data_path = data_path
        self.label_list = label_list
        self.l2i = l2i
        self.first_n_byte = first_n_byte

    def __len__(self):
        return len(self.fp_list)

    @staticmethod
    def represent_bytes(bytes_str):
        """
        :param bytes_str: string of bytes, i.e. two characters, each is one of 0-f.
        :return: integer, number between 0 to 257.
        """
        if bytes_str == '??':  # ignore those signs
            return 0
        return int(bytes_str, 16) + 1

    def __getitem__(self, idx):
        with open(self.data_path + self.fp_list[idx] + '.bytes', 'r') as f:
            tmp = []
            for line in f:
                line = line.split()
                line.pop(0)  # ignore address

                line = map(ExeDataset.represent_bytes, line)
                tmp.extend(line)

            # padding with zeroes such that all files will be of the same size
            tmp = tmp + [0] * (self.first_n_byte - len(tmp))

        return np.array(tmp), np.array([self.l2i[int(self.label_list[idx])]])
