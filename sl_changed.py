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











def decompose_tool(func, T, tag, tag1, tag2):
    tmp_tag = T.tags
    tmp = tmp_tag.index(tag)
    T.transpose(tmp_tag[:tmp] + tmp_tag[tmp+1:] + [tag])
    q, r = func(T, len(T.tags)-1, tag1, tag2)
    T.transpose(tmp_tag)
    return q,r

def unitarilize(node_list, target_tag, counter_tag):
    for i in range(len(node_list)-1):
        node_list[i] , r = \
            decompose_tool(Node.qr, node_list[i], target_tag, target_tag, counter_tag)
        node_list[i+1] = \
            Node.contract(node_list[i+1], [counter_tag], r, [target_tag])

##generate 4 directions approximate lines
#from "l" to "r"
sub_l = lattice(node_approx, L1, L2-1, "udrr")
for i in range(L1):
    for j in range(L2):
        latt[i][j].normf = False

for j in range(L2-1):
    ##down unitarilization
    tmp = []
    for i in range(L1):
        tmp = Node.copy(sub_l[i][j]) + tmp
    unitarilize(tmp, "d", "u")
    for i in range(L1):
        sub_l[i][j].replace(tmp[L1-i-1])
    ##generate "side"
    side = []
    for i in range(L1-1, 0, -1):
        tmp = Node.contract(latt[i][j], ["r"], sub_l[i][j], ["r_up"], \
                            {"u":"up","d":"coun_up", "r":"r_up"}, {"u":"new","d":"coun_down"})
        tmp = Node.contract(tmp, ["r_down", "p"], latt[i][j], ["r", "p"], \
                            {}, {"u":"down", "d":"coun_down", "l":"l'"})
    ##generate approximate sub_l
    i = 0
    step = 1
    for _ in range(2 * L1 - 1):
        pass


