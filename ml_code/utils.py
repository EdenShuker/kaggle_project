from collections import Counter
from csv import DictReader
from os import listdir
from os.path import isfile, join

import capstone
import pefile

UNK = '_UNK_'

BYTES_END = 'bytes'
ASM_END = 'asm'
DLL_END = 'dll'
SEGMENT_END = 'segments'


def get_ngrams_set_of(f_name, n=4):
    """
    :param f_name: file name.
    :param n: num of grams to concat.
    :return: set of ngrams of the given file.
    """
    path_to_file = "%s.%s" % (f_name, BYTES_END)
    one_list = []
    with open(path_to_file, 'rb') as f:
        for line in f:
            # append bytes to list
            line = line.rstrip().split(" ")
            line.pop(0)  # ignore address
            one_list += line

    # array holds all 4 grams opcodes (array of strings) . use sliding window.
    grams_list = [''.join(one_list[i:i + n]) for i in xrange(len(one_list) - n + 1)]

    # create a set of ngrams out of the ngrams
    ngrams_set = set()
    ngrams_set.update(grams_list)
    return ngrams_set


def get_files_from_dir(dirpath, ending):
    """
    :param dirpath: path to directory.
    :param ending: file-ending, the type of files you want to get.
    :return: list of files names that has the given ending.
    """
    end_len = len(ending)
    files = [f[:-end_len] for f in listdir(dirpath) if isfile(join(dirpath, f)) and f.endswith(ending)]
    return files


def produce_data_file_on_segments(dll_filename, dirpath='/media/user/New Volume/benign'):
    """
    Create a file that will provide information about the segments
        of the given file (name of segment and number of lines in it).
    The output file will have the same name as the input-file but instead, with '.segments' ending.

    :param dll_filename: name of dll file.
    """
    file_name = dll_filename.rsplit('/', 1)[1]
    try:
        pe = pefile.PE('%s/%s.%s' % (dirpath, file_name, DLL_END))
    except Exception as e:
        print 'Error with pefile on file: %s/%s.%s' % (dirpath, file_name, DLL_END)
        print e.message
        exit(0)
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    full_path = '%s/%s.%s' % ('ml_code/segments_data', file_name, SEGMENT_END)
    if not isfile(full_path):
        with open(full_path, 'w') as f:
            for section in pe.sections:
                code = section.get_data()
                first_instruction_address = section.PointerToRawData
                num_lines_in_section = 0
                for i in md.disasm(code, first_instruction_address):
                    num_lines_in_section += 1
                # for each section write in the file the name and number of lines in that section
                f.write('%s:%i\n' % (section.Name.strip('\0'), num_lines_in_section))


def count_seg_counts(f_name, seg_set, dirpath='ml_code/segments_data'):
    """
    :param f_name: name of file.
    :param seg_set: set of segments-names.
    :return: dict that maps segment-name to number of lines in that segment in the given file.
    """
    name = f_name.rsplit('/', 1)[1]
    seg_counter = Counter()
    num_unks = 0  # number of unknown segments

    filepath_without_ending = '%s.' % (f_name)
    path_to_file = filepath_without_ending + ASM_END
    if isfile(path_to_file):  # can use .asm file
        mode = ASM_END
    else:  # has no .asm file, so parse the dll file and extract info about segments
        produce_data_file_on_segments(f_name)
        path_to_file = dirpath + '/' + name + '.' + SEGMENT_END
        mode = SEGMENT_END

    with open(path_to_file, 'rb') as f:
        for line in f:
            seg_name, rest = line.split(':', 1)
            if seg_name not in seg_set:  # if it is unknown segment (was not in train set) mark it as UNK
                seg_name = UNK
                num_unks += 1

            if mode == ASM_END:  # in .asm file, the segment name appears for each line in it
                val_to_add = 1
            else:  # in .segments file, the segment name appears alongside the number of lines in it
                val_to_add = int(rest)
            seg_counter[seg_name] += val_to_add
    if num_unks > 0:  # for UNK segments, take the average number of lines
        seg_counter[UNK] = int(seg_counter[UNK] / num_unks)
    return seg_counter


def read_csv(filepath, key1, key2):  # TODO documentation
    """
    :param filepath: path to f2l-file (train_labels_filtered.csv) .
    :param key1:
    :param key2:
    :return: file-to-label dict.
    """
    k2v_dict = dict()
    csv_dict = DictReader(open(filepath))
    for row in csv_dict:
        key = row[key1]
        val = row[key2]
        k2v_dict[key] = val
    return k2v_dict
