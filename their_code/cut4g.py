import sys
import pickle

newc = pickle.load(open('cmd4g.p'))
nx = {}
c = 0
for i in newc:
    if newc[i] > 100:
        c += 1
        nx[i] = newc[i]
print c, len(newc)
pickle.dump(nx, open('cutcmd4g.p', 'w'))
