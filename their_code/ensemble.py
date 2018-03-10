import pandas as pd

s1 = pd.read_csv('model1.csv', index_col=0)
s2 = pd.read_csv('model2.csv', index_col=0)
s3 = pd.read_csv('model3.csv', index_col=0)
for i in s1.columns.values:
    s1[i] = s1[i] ** 0.1 * s2[i] ** 0.4 * s3[i] * 0.5
s1.to_csv('ensemble.csv')
