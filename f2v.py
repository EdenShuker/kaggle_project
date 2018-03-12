import pickle
import sys
from csv import DictReader

import utils

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


def main():
    """
    Parameters to main:
        dirpath f2l_filepath f2v_filepath
        # dirpath - path to directory where the files are in.
        # files_filepath - path to a .csv file, contains a column of 'Id' with file-name in each row.
        # f2v_filepath - path to f2v file. the name of the file to create.
                         will be in format of 'filename<tab>vector<EOL>'
    """
    args = sys.argv[1:]
    dirpath = args[0]
    files_filepath = args[1]
    f2v_filepath = args[2]

    # extract names of files
    file_list = []
    csv_dict = DictReader(open(files_filepath))
    for row in csv_dict:
        file_list.append(row['Id'])

    create_file_file2vec(dirpath, file_list, f2v_filepath)
    print 'done created %s' % f2v_filepath


if __name__ == '__main__':
    main()
