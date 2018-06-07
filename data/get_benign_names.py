from collections import Counter
from csv import DictReader


def main(filepath, key2):
    k2v_dict = Counter()
    csv_dict = DictReader(open(filepath))
    for row in csv_dict:
        val = row[key2]
        k2v_dict[val] += 1
    for k in k2v_dict:
        print k, k2v_dict[k]

if __name__ == '__main__':
    main('trainLabels.csv', 'Class')
    # 1, 9, 8
