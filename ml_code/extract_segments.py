import pickle
from time import time

import pefile

from ml_code import utils


def get_segment_set_of(dirpath, train_set_path):
    """
    :param dirpath: path to directory.
    :return: set of segments-names extracted from all the files in the given directory.
    """
    seg_set = set()
    train_set = utils.get_f2l_dict(train_set_path).viewkeys()
    # segments from .asm files
    ASM_END = utils.ASM_END
    asm_files = utils.get_files_from_dir(dirpath, '.' + ASM_END)  # get list of .asm files
    for asm_f in asm_files:
        if asm_f in train_set:
            with open('%s/%s.%s' % (dirpath, asm_f, ASM_END)) as f:
                for line in f:
                    segment_name = line.split(':', 1)[0]
                    seg_set.add(segment_name.rstrip('\x00'))

    # segments from .dll files
    DLL_END = utils.DLL_END
    dll_files = utils.get_files_from_dir(dirpath, '.' + DLL_END)  # get list of .dll files
    for dll_f in dll_files:
        if dll_f in train_set:
            try:
                pe = pefile.PE('%s/%s.%s' % (dirpath, dll_f, DLL_END))
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
    segments = get_segment_set_of('data/files', 'data/train_set.csv')
    segments.add(utils.UNK)
    print segments
    pickle.dump(segments, open('features/segments_features', 'w'))

    print 'time to run:', time() - t0
