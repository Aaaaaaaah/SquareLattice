# -*- coding: utf-8 -*-

import numpy as np
from node import Node

L1 = 4
L2 = 4
D = 4

#   u
# l   r
#   d

direct_dict = {"u":(lambda i,j:i==0), \
               "d":(lambda i,j:i==L1-1), \
               "l":(lambda i,j:j==0), \
               "r":(lambda i,j:j==L2-1)}

def _node(s):
    l = [D]*len(s)
    l.append(2)
    return Node(list(s)+["p"],l)

def node_in_lattice(s,i,j):
    for x in s:
        if direct_dict[x](i,j):
            s = s.replace(x,"")
    return _node(s)

def node_approx(s, i, j):
    for x in s:
        if direct_dict[x](i,j):
            s = s.replace(x,"")
    s = list(s)
    for x in s:
        if x in s[s.index(x)+1:]:
            s[s.index(x)+1] = s[s.index(x)+1] + "_down"
            s[s.index(x)] = s[s.index(x)] + "_up"
            break
    l = [D]*len(s)
    return Node(s, l)

def lattice(func, n, m, s):
    return [[func(s, i, j) for j in range(m)] for i in range(n)]

latt = lattice(node_in_lattice, L1, L2, "udrl")

ep = 0.1

H = np.reshape([[0.25,0,0,0],[0,-0.25,0.5,0],[0,0.5,-0.25,0],[0,0,0,0.25]],[2,2,2,2])
I = np.reshape(np.identity(4),[2,2,2,2])
expH = I - 4.*ep*H

for t in range(10):
    for i in range(0,L1):
        for j in range(0,L2-1,2):
            Node.update(latt[i][j],latt[i][j+1],"r","l","p","p",expH)
    for i in range(0,L1):
        for j in range(1,L2-1,2):
            Node.update(latt[i][j],latt[i][j+1],"r","l","p","p",expH)
    for j in range(0,L2):
        for i in range(0,L1-1,2):
            Node.update(latt[i][j],latt[i+1][j],"d","u","p","p",expH)
    for j in range(0,L2):
        for i in range(1,L1-1,2):
            Node.update(latt[i][j],latt[i+1][j],"d","u","p","p",expH)

print(latt[1][1].envs)

##generate 4 directions approximate lines
#from "l" to "r"
sub_l = lattice(node_approx, L1, L2-1, "udrr")


