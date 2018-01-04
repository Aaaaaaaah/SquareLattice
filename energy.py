# -*- coding: utf-8 -*-

"""
    polished from contract_two_rows.py
    try to support the energy calculation of square lattice
"""

import numpy as np
from node import Node
from tool import decompose_tool, very_simple_contract, unitarilize

#hat
hat = [Node(["phy"], [2], data=np.array([1,0])), Node(["phy"], [2], data=np.array([0,1]))]

class square_lattice(object):
    """
        tensor_array
        lattice_type
        redu_tensor redutensor[i][j][0 or 1]
    """
    def __init__(self, arr, rows, cols, H):
        self.tensor_array = [[Node.copy(j) for j in i] for i in arr]
        self.redu_tensor = [[[Node.contract(j,["phy"],hat[k], ["phy"]) \
                              for k in range(2)] for j in i] for i in arr]
        self.Hamilton = H.copy()

    def contract_two_row(psi0, operator, left="l", up="u", down="d", right="r"):
        """
            psi0: left, up, right
            operator: left, up, down, right
        """
        delta = 1e-6
        L = len(psi0)
        ##disable normf
        for i in psi0 + operator:
            i.normf = False
        ##unitarilize
        psi_new = [Node.copy(i) for i in psi0]
        unitarilize(psi_new, left, right)
        ##initiate side
        tmp = Node.contract(psi0[L-1], [up], operator[L-1], [down], {left:"down"}, {left:"mid"})
        tmp = Node.contract(tmp, [up], psi_new[L-1], [up], {}, {left:"up"})
        side = [tmp]
        for i in range(L-2, 0, -1):
            tmp = Node.contract(tmp, ["down"], psi0[i], [right], {}, {left:"down"})
            tmp = Node.contract(tmp, ["mid", up], operator[i], [right, down], {}, {left:"mid"})
            tmp = Node.contract(tmp, ["up", up], psi_new[i], [right, up], {}, {left:"up"})
            side = [tmp] + side
        side = [None] + side
        ##target energy
        tmp = [[Node.copy(i).rename_leg({up:down}) for i in psi0], \
               [Node.copy(i).rename_leg({up:down, down:up}) for i in operator], operator, psi0]
        energy0 = very_simple_contract(tmp, 4, L, up, down, left, right)
        ##main part
        dir = 1
        dir_dict = {1:right, -1:left}
        pos = 0
        energy1 = very_simple_contract([[Node.copy(i).rename_leg({up:down}) for i in psi_new], \
                                        psi_new], 2, L, up, down, left, right)
        print(energy0)
        while (abs(energy1-energy0)>abs(energy0)*delta):
            print(pos, energy1)
            in_range = (pos-dir) in range(L)
            if in_range:
                tmp = Node.contract(side[pos-dir], ["down"], psi0[pos], [dir_dict[-dir]], {}, {dir_dict[dir]:"down"})
                tmp = Node.contract(tmp, ["mid", up], operator[pos], [dir_dict[-dir], down], {}, {dir_dict[dir]:"mid"})
            else:
                tmp = Node.copy(psi0[pos]).rename_leg({dir_dict[dir]:"down"})
                tmp = Node.contract(tmp, [up], operator[pos], [down], {}, {dir_dict[dir]:"mid"})
            psi_new[pos] = Node.contract(tmp, ["mid", "down"], side[pos+dir], ["mid", "down"], {"up":dir_dict[-dir]} \
                                if in_range else {}, {"up":dir_dict[dir]})
            psi_new[pos],r = decompose_tool(Node.qr, psi_new[pos], dir_dict[dir], dir_dict[dir], dir_dict[-dir])
            psi_new[pos+dir] = Node.contract(r, [dir_dict[dir]], psi_new[pos+dir], [dir_dict[-dir]])
            side[pos] = Node.contract(tmp, ["up", up] if in_range else [up], psi_new[pos], \
                                      [dir_dict[-dir], up] if in_range else [up], {}, {dir_dict[dir]:"up"})
            pos += dir
            if pos in [0, L-1]:
                dir = -dir
            energy1 = very_simple_contract([[Node.copy(i).rename_leg({up:down}) for i in psi_new], \
                                            psi_new], 2, L, up, down, left, right)
        return psi_new
