import sys
import pickle

newc = pickle.load(open('cmd3g.p'))
nx = {}
c = 0
for i in newc:
    if newc[i] > 100:
        c += 1
        nx[i] = newc[i]
print c, len(nx)
pickle.dump(nx, open('cutcmd3g.p', 'w'))
