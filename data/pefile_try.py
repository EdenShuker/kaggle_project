import pefile
from capstone import *

import utils


def do_something(dirpath, ending):
    benign_files = utils.get_files_from_dir(dirpath, ending)
    some_file = benign_files[0]

    pe = pefile.PE('%s/%s%s' % (dirpath, some_file, ending))

    md = Cs(CS_ARCH_X86, CS_MODE_64)
    for section in pe.sections[:2]:
        print section.Name.strip('\0')
        # section attr - VirtualAddress, PointerToRawData
        code = section.get_data()
        first_instruction_address = section.PointerToRawData
        for i in md.disasm(code, first_instruction_address):
            print '0x%x:\t%s\t%s' % (i.address, i.mnemonic, i.op_str)
        print '\n'


if __name__ == '__main__':
    do_something('files', '.dll')
