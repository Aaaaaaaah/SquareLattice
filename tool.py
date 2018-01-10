# -*- coding: utf-8 -*-

"""
    put some relevent functions here
    these things should be handled more seriously...
"""

import numpy as np
from node import Node

def decompose_tool(func, T, tag, tag1, tag2):
    tmp_tag = T.tags
    tmp = tmp_tag.index(tag)
    T.transpose(tmp_tag[:tmp] + tmp_tag[tmp+1:] + [tag])
    q, r = func(T, len(T.tags)-1, tag1, tag2)
    T.transpose(tmp_tag)
    return q, r

def very_simple_contract(sq_latt, n, m, u="u", d="d", l="l", r="r"):
    #   u
    # l   r
    #   d
    ans = Node([], [])
    for j in range(m):
        for i in range(n):
            tag1 = []
            tag2 = []
            if i!=0:
                tag1 += [d]
                tag2 += [u]
            if j!=0:
                tag1 += [str(i)]
                tag2 += [l]
            if j==(m-1):
                dict2 = {}
            else:
                dict2 = {r:str(i)}
            ans = Node.contract(ans, tag1, sq_latt[i][j], tag2, {}, dict2)
    return ans.data.copy()

def unitarilize(node_list, target_tag, counter_tag):
    for i in range(len(node_list)-1, 0, -1):
        node_list[i] , r = \
            decompose_tool(Node.qr, node_list[i], target_tag, target_tag, counter_tag)
        node_list[i-1] = \
            Node.contract(node_list[i-1], [counter_tag], r, [target_tag])

def attempt_step(spins):
    n = len(spins)
    m = len(spins[0])
    new_n = random.randint(n)
    new_m = random.randint(m)
    new_spins = [[j.copy() for j in i]for i in spins]
    new_spins[new_n][new_m] = 1 - new_spins[new_n][new_m]
    return new_spins
