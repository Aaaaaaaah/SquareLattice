# -*- coding: utf-8 -*-
# 4 x 4 OBD

import numpy as np
from node import Node

L1 = 4
L2 = 4
D = 4

#   u
# l   r
#   d

def _node(s):
    l = [D]*len(s)
    l.append(2)
    return Node(list(s)+["p"],l)

def node_in_lattice(i,j):
    if i == 0:
        if j == 0:
            return _node("dr")
        elif j != L2-1:
            return _node("ldr")
        else:
            return _node("ld")
    elif i != L1:
        if j == 0:
            return _node("urd")
        elif j != L2-1:
            return _node("ulrd")
        else:
            return _node("uld")
    else:
        if j == 0:
            return _node("ur")
        elif j !=  L2-1:
            return _node("ulr")
        else:
            return _node("ul")

lattice = [[node_in_lattice(i,j) for j in range(L2)] for i in range(L1)]

ep = 0.1

H = np.reshape([[0.25,0,0,0],[0,-0.25,0.5,0],[0,0.5,-0.25,0],[0,0,0,0.25]],[2,2,2,2])
I = np.reshape(np.identity(4),[2,2,2,2])
expH = I - 4.*ep*H

for t in range(100):
    print t
    for i in range(0,L1):
        for j in range(0,L2-1,2):
            Node.update(lattice[i][j],lattice[i][j+1],"r","l","p","p",expH)
    for i in range(0,L1):
        for j in range(1,L2-1,2):
            Node.update(lattice[i][j],lattice[i][j+1],"r","l","p","p",expH)
    for j in range(0,L2):
        for i in range(0,L1-1,2):
            Node.update(lattice[i][j],lattice[i+1][j],"d","u","p","p",expH)
    for j in range(0,L2):
        for i in range(1,L1-1,2):
            Node.update(lattice[i][j],lattice[i+1][j],"d","u","p","p",expH)

print lattice[1][1].envs
