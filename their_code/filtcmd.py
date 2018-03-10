import pickle

cmd = pickle.load(open('newcmd.p'))
newc = {}
for c in cmd:
    if '_' in c or c[0] in '?1234567890ABCDEF':
        continue
    else:
        # print c,cmd[c]
        newc[c] = cmd[c]
print newc
pickle.dump(newc, open('newc.p', 'w'))
