from csv import DictReader

label2dir = {'5': 'Benign', '1': 'Gatak', '2': 'Kelihos', '3': 'Kryptik', '4': 'Lollipop', '6': 'Ramnit', '7': 'Simda',
             '8': 'Vundo', '9': 'Zbot'}

file2label = open('allLabels.csv')
with open('files2label.csv', 'w') as f:
    for row in file2label:
        file_name, label = row.split()
        f.write('{}/{},{}\n'.format(label2dir[label], file_name, label))
f.close()
