import sys
import pickle

newc = pickle.load(open('cmd3g.p'))
nx = {}
c = 0
for i in newc:
    if newc[i] > 10000:
        c += 1
        nx[i] = newc[i]
print c, len(newc)
pickle.dump(nx, open('cutcmd3g_for_4g.p', 'w'))
