import pickle
from time import time
import pefile
import utils


def get_segment_set_of(dirpath, train_set_path):
    """
    :param dirpath: path to directory.
    :param train_set_path: path to trainLabels.csv .
    :return: set of segments-names extracted from all the files in the given directory.
    """
    seg_set = set()
    train_set = utils.read_csv(train_set_path, 'Id', 'Class').viewkeys()
    # segments from .asm files
    ASM_END = utils.ASM_END
    asm_files = utils.get_files_from_dir(dirpath, '.' + ASM_END)  # get list of .asm files
    for asm_f in asm_files:
        full_path = dirpath + '/' + asm_f
        if full_path in train_set:
            with open('%s.%s' % (full_path, ASM_END)) as f:
                for line in f:
                    segment_name = line.split(':', 1)[0]
                    seg_set.add(segment_name.rstrip('\x00'))

    # segments from .dll files
    DLL_END = utils.DLL_END
    # TODO in ASAFIS the dll_files list is empty because the .bytes and .dll files are in different dirs,
    # TODO thus the dirpath here is of the .bytes dir but needed .dll dirpath
    dll_files = utils.get_files_from_dir(dirpath, '.' + DLL_END)  # get list of .dll files
    for dll_f in dll_files:
        full_path = dirpath + '/' + dll_f
        if full_path in train_set:
            try:
                pe = pefile.PE('%s.%s' % (full_path, DLL_END))
            except Exception as e:
                print 'Error with pefile on file: %s' % dll_f
                print e.message
                continue
            for section in pe.sections:
                seg_set.add(section.Name.rstrip('\x00'))
    return seg_set


if __name__ == '__main__':
    t0 = time()

    # TODO add the opportunity to pass this path as args to main
    segments1 = get_segment_set_of('/home/tamir/PycharmProjects/kaggle_project_new/data/benign', 'data/train_set.csv')
    segments2 = get_segment_set_of('/home/tamir/PycharmProjects/kaggle_project_new/data/malware', 'data/train_set.csv')
    segments1.update(segments2)
    segments1.add(utils.UNK)
    print segments1
    pickle.dump(segments1, open('ml_code/features/segments_features', 'w'))

    print 'time to run:', time() - t0
