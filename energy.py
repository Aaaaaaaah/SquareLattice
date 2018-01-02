# -*- coding: utf-8 -*-

"""
    polished from contract_two_rows.py
    try to support the energy calculation of square lattice
"""

import numpy as np
from node import Node
from tool import decompose_tool, very_simple_contract, unitarilize

class square_lattice(object):
    """
        tensor_array
        lattice_type
    """
    def __init__(self, arr, rows, cols):
        o
        self.tensor_array = [i.copy() for i in arr]

##initiate side
tmp = Node.contract(operator[-1], ["phyd"], psi0[-1], ["phy"])
tmp.rename_leg({"l":"down","ol":"mid"})
side = [Node.contract(tmp, ["phyu"], psi_new[-1], ["phy"])]
side[0].rename_leg({"l":"up"})
for i in range(L-2, 0, -1):
    tmp = Node.contract(operator[i], ["phyd"], psi0[i], ["phy"])
    tmp.rename_leg({"l":"down", "ol":"mid"})
    tmp = Node.contract(tmp, ["r", "or"], side[0], ["down", "mid"])
    tmp = Node.contract(tmp, ["phyu", "up"], psi_new[i], ["phy", "r"])
    tmp.rename_leg({"l":"up"})
    side = [tmp] + side
side = [Node(["up", "mid", "down"], [D2, D1, D1])] + side

##main part
for _ in range(1):
    #from left to right
    for i in range(L-1):
        tmp = Node.contract(psi0[i], ["phy"], operator[i], ["phyd"])
        tmp.rename_leg({"phyu":"phy"})
        if i is not 0 :
            tmp = Node.contract(tmp, ["ol", "l"], side[i-1], ["mid", "down"])
            tmp.rename_leg({"up":"l"})
        side[i] = Node.copy(tmp)
        side[i].rename_leg({"r":"down", "or":"mid"})
        tmp = Node.contract(tmp, ["or", "r"], side[i+1], ["mid", "down"])
        tmp.rename_leg({"up":"r"})
        psi_new[i] , r = decompose_tool(Node.qr, tmp, "r", "r", "l")
        if i is not 0:
            side[i] = Node.contract(side[i], ["phy", "l"], psi_new[i], ["phy", "l"])
        else:
            side[i] = Node.contract(side[i], ["phy"], psi_new[i], ["phy"])
        side[i].rename_leg({"r":"up"})
    #from right to left
    for i in range(L-1, 0, -1):
        tmp = Node.contract(psi0[i], ["phy"], operator[i], ["phyd"])
        tmp.rename_leg({"phyu":"phy"})
        if i is not L-1 :
            tmp = Node.contract(tmp, ["or", "r"], side[i+1], ["mid", "down"])
            tmp.rename_leg({"up":"r"})
        side[i] = Node.copy(tmp)
        side[i].rename_leg({"l":"down", "ol":"mid"})
        tmp = Node.contract(tmp, ["ol", "l"], side[i-1], ["mid", "down"])
        tmp.rename_leg({"up":"l"})
        psi_new[i] , r = decompose_tool(Node.qr, tmp, "l", "l", "r")
        if i is not L-1:
            side[i] = Node.contract(side[i], ["phy", "r"], psi_new[i], ["phy", "r"])
        else:
            side[i] = Node.contract(side[i], ["phy"], psi_new[i], ["phy"])
        side[i].rename_leg({"l":"up"})
tmp = Node.contract(psi0[0], ["phy"], operator[0], ["phyd"])
tmp.rename_leg({"phyu":"phy"})
tmp = Node.contract(tmp, ["or", "r"], side[1], ["mid", "down"])
tmp.rename_leg({"up":"r"})
psi_new[0] = Node.copy(tmp)
