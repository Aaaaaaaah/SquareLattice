import numpy as np
from node import Node

L1 = 5
L2 = 6
D = 4

#   u
# l   r
#   d

# +-+ +-+ +-+
# | | | | | |
# + +-+ +-+ +
# | | | | | |
# +-+ +-+ +-+
# | | | | | |
# + +-+ +-+ +
# | | | | | |
# +-+ +-+ +-+

def _node(s):
    l = [D]*len(s)
    l.append(2)
    return Node(list(s)+["p"],l)

def node_in_lattice(i,j):
    if (i-j)%2 == 0:
        s = "udr"
    else:
        s = "udl"
    if i == 0:
        s = s.replace("u","")
    if i == L1-1:
        s = s.replace("d","")
    if j == 0:
        s = s.replace("l","")
    if j == L2-1:
        s = s.replace("r","")
    return _node(s)

lattice = [[node_in_lattice(i,j) for j in range(L2)] for i in range(L1)]

ep = 0.1

H = np.reshape([[0.25,0,0,0],[0,-0.25,0.5,0],[0,0.5,-0.25,0],[0,0,0,0.25]],[2,2,2,2])
I = np.reshape(np.identity(4),[2,2,2,2])
expH = lambda x:I - x*ep*H

for t in range(10):
    print(t)
    for i in range(0,L1):
        for j in range(i%2,L2-1,2):
            Node.update(lattice[i][j],lattice[i][j+1],"r","l","p","p",expH(1))
    for j in range(0,L2):
        for i in range(j%2,L1-1,2):
            Node.update(lattice[i][j],lattice[i+1][j],"d","u","p","p",expH(2))
    for j in range(0,L2):
        for i in range((j+1)%2,L1-1,2):
            Node.update(lattice[i][j],lattice[i+1][j],"d","u","p","p",expH(3))

print(lattice[1][1].envs)
