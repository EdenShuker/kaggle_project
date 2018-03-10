import sys
import pickle

##########################################################
# usage
# pypy find_2g.py xid_train.p ../../data/train 

# xid_train.p is a list like ['loIP1tiwELF9YNZQjSUO',''....] to specify
# the order of samples in traing data
# ../../data/train is the path of original train data
##########################################################
xid_name = sys.argv[1]
data_path = sys.argv[2]

xid = pickle.load(open(xid_name))  # xid_train.p or xid_test.p

newc = pickle.load(open('newc.p'))
cmd2g = {}
for i in newc:
    for j in newc:
        cmd2g[(i, j)] = 0
print newc

for c, f in enumerate(xid):  # (files[len(files)/10*a1:len(files)/10*a2]):
    count = {}
    for i in cmd2g:
        count[i] = 0
    fo = open(data_path + '/' + f + '.asm')
    tot = 0
    a = -1
    b = -1
    for line in fo:
        xx = line.split()
        for x in xx:
            if x in newc:

                a = b
                b = x
                if (a, b) in cmd2g:
                    count[(a, b)] += 1
                    tot += 1
    #                     print (b,a)
    fo.close()
    if c % 10 == 0:
        print c * 1.0 / len(xid), tot
    for i in cmd2g:
        cmd2g[i] = count[i] + cmd2g[i]
    del count

import pickle

cmd2gx = {}
for i in cmd2g:
    if cmd2g[i] > 10:
        cmd2gx[i] = cmd2g[i]
print len(cmd2gx)
pickle.dump(cmd2gx, open('cmd2g.p', 'w'))
