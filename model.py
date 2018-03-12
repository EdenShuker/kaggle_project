import pickle
from csv import DictReader
from time import time
import sys

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import utils

CREATE_F2V_FILE = '-create-f2v'
SAVE_MODEL = '-save'
SHOW_CONFUSION_MAT = '-show-matrix'
TRAIN = '-train'
TEST = '-test'
LOAD_MODEL = '-load'
OUTPUT_FILE_FLAG = '-out-file'

# load feature
ngrams_features_list = pickle.load(open('features/ngrams_features'))
segments_features_set = pickle.load(open('features/segments_features'))


def represent_file_as_vector(dirpath, filename):
    """
    :param dirpath: path of directory that the given file is in.
    :param filename: name of file, with the extension(= .bytes or .asm) .
    :return: vector of features that represents the given file.
    """
    vec = []

    # ngrams
    curr_ngrams_set = utils.get_ngrams_set_of(dirpath, filename, n=4)
    for feature in ngrams_features_list:
        # TODO current - boolean of 'is ngram in file', optional - how many time ngrams in file
        if feature in curr_ngrams_set:
            vec.append(1)
        else:
            vec.append(0)

    # segments
    seg_counter = utils.count_seg_counts(dirpath, filename, segments_features_set)
    for seg_name in segments_features_set:
        if seg_name in seg_counter:
            vec.append(seg_counter[seg_name])
        else:
            vec.append(0)

    return vec


def create_file_file2vec(dirpath, files_list, f2v_name):
    """
    :param dirpath: path to directory that the given files are in.
    :param files_list: list of files-names.
    :param f2v_name: output file, will contain file-name and the vector that represents it.
        Format of file: filename<\t>vec
    """
    # TODO change the format of each vector only in the f2v.file
    #      for each vector, for ngrams save only the indexes and for segments as regular
    #      when reading the f2v.file and the model needs to train parse it to a regular vector by starting
    #      with zeroes and then modify the vector according to the indexes
    with open(f2v_name, 'w') as f:
        for f_name in files_list:
            vec = represent_file_as_vector(dirpath, f_name)  # represent each file as a vector
            vec = map(lambda x: str(x), vec)
            f.write(f_name + '\t' + ' '.join(vec) + '\n')


def get_data_and_labels(f2l_filepath, f2v_filepath):
    """
    :param f2l_filepath: path to file-to-label file (train_set.csv / test_set.csv) .
    :param f2v_filepath: path to file-to-vector file.
    :return: matrix and array of labels, the ith label there is connected to the ith vector in matrix.
    """
    matrix, labels = [], []

    # create file-to-label dict
    f2l_dict = utils.get_f2l_dict(f2l_filepath)
    with open(f2v_filepath, 'r') as f:
        for line in f:  # for each file find the vector that representing him
            filename, vec = line.split('\t')
            if filename in f2l_dict.viewkeys():
                vec = map(lambda (x): int(x), vec.split(' '))
                matrix.append(vec)
                labels.append(f2l_dict[filename])

    matrix, labels = np.array(matrix), np.array(labels)
    return matrix, labels


def get_data(f2v_filepath):
    """
    :param f2v_filepath: path to file-to-vector file.
    :return: matrix which contains all the vectors inside the file.
    """
    matrix = []
    with open(f2v_filepath) as f:
        for line in f:
            _, vec = line.split('\t')
            vec = map(lambda x: int(x), vec.split(' '))
            matrix.append(vec)
    return matrix


class CodeModel(object):
    """
    Model that works with code-files, predict the type of file out of 10 possible type.
    """

    def __init__(self, lr=0.1, n_estimators=30, max_depth=5, min_child_weight=1,
                 gamma=0, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, seed=27):
        # TODO need to tune the parameters
        self.model = xgb.XGBClassifier(learning_rate=lr,
                                       n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       min_child_weight=min_child_weight,
                                       gamma=gamma,
                                       subsample=subsample,
                                       colsample_bytree=colsample_bytree,
                                       scale_pos_weight=scale_pos_weight,
                                       objective='multi:softprob',
                                       seed=seed)

    def predict_on(self, matrix):
        """
        :param matrix: each row in it is a vector that represents some file.
        :return: list of labels, the ith-label is connected to the ith-vector in matrix.
        """
        preds = self.model.predict(matrix)
        # return [round(val) for val in preds]
        return preds

    def predict_and_accuracy_on(self, matrix, labels, show_confusion_matrix=False):
        """
        :param matrix: each row in it is a vector that represents some file.
        :param labels: list of labels, the ith-label is connected to the ith-vector in matrix.
        :param show_confusion_matrix: boolean, determine if to show to the user confusion matrix.
        :return:
        """
        # predict and find accuracy
        preds = self.predict_on(matrix)
        acc = accuracy_score(labels, preds)
        print 'accuracy %0.2f%%' % (acc * 100.0)

        # confusion matrix
        if show_confusion_matrix:
            print confusion_matrix(labels, preds)

    def train_on(self, train_matrix, labels, model_name=None):
        """
        Fit the model on the train-set and check its performance on dev-set, can save the model after training.
        :param train_matrix: data set.
        :param labels: list of labels, ith-label is connected to the ith-item in matrix.
        :param model_name: string if needed to save the model,
            the saved model will be in file named by this string, None as default for not saving the model.
        """
        # fit model to training data
        self.model.fit(train_matrix, labels, eval_metric='mlogloss')

        # save model if needed
        if model_name:
            self.save_model(model_name)
            print 'saved model'

    def save_model(self, filename):
        """ save the current model in a file, can be loaded from that file later. """
        pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load_from(filename):
        """ load a model from file """
        model = pickle.load(open(filename, 'rb'))
        return model


def main():
    """
    parameters to main (can play with the options):
    [-load model_name]
    [-train train_csv_file f2v_file]
    [-save new_model_name]
    [-test test_csv_file f2v_file output_file]
    """
    t0 = time()
    args = sys.argv[1:]

    # create model
    if LOAD_MODEL in args:
        given_model_file = args[args.index(LOAD_MODEL) + 1]
        model = CodeModel.load_from(given_model_file)
        print 'loaded model'
    else:
        model = CodeModel()

    # train model
    if TRAIN in args:
        i = args.index(TRAIN)
        csv_filepath = args[i + 1]
        train_f2v_filepath = args[i + 2]
        train, y_train = get_data_and_labels(csv_filepath, train_f2v_filepath)

        # save model
        model_name = None
        if SAVE_MODEL in args:
            model_name = args[args.index(SAVE_MODEL) + 1]

        model.train_on(train, y_train, model_name)
        print 'trained on data'

    # blind test
    if TEST in args:
        i = args.index(TEST)
        csv_filepath = args[i + 1]
        test_f2v_filepath = args[i + 2]
        output_filepath = args[i + 3]

        matrix = get_data(test_f2v_filepath)
        preds = model.predict_on(matrix)

        # write output file
        with open(output_filepath, 'w') as out_file:
            out_file.write('Id,Class\n')
            input_file = DictReader(open(csv_filepath))
            for row, label in zip(input_file, preds):
                out_file.write('%s,%s\n' % (row['Id'], label))
        print 'done predicting on test'

    print 'time to run model:', time() - t0


if __name__ == '__main__':
    """
    parameters to main:
    [-train csv_file f2v_file] [-save new_model_name] [-load model_name] [-test f2v_file output_file]
    -train  data/train_set.csv f2v/train.f2v -save first.model
    """
    main()
