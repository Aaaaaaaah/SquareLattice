import numpy as np
from node import Node

L = 10
d = 2
D = 4

#from right to leftt
#contract two rows into one row

##generate psi0(left, up, right)
psi0 = [Node(["phy", "r"], [d, D])] +[Node(["l", "phy", "r"], [D, d, D]) for _ in range(L-2)] + [Node(["l", "phy"], [D, d])]

##generate Operator(left, up, down, right)
Operator = [Node(["phyu", "phyd", "or"], [d, d, D])] + [Node(["ol", "phyu", "phyd", "or"], [D, d, d, D]) for _ in range(L-2)] + [Node(["ol", "phyu", "phyd"], [D, d, d])]

##psi_new = psi0
psi_new = [Node.copy(i) for i in psi0]

##right unitarinalize
for i in range(L-1, 0, -1):
    psi_new[i].data , r = Node.qr(psi_new[i], ["l"])
    psi_new[i-1].matrix_multiply("r", r, 1)

##initiate side
tmp = Node.contract(operator[-1], ["phyd"], psi0[-1], ["phy"])
tmp.rename_leg({"l":"down","ol":"mid"})
side = [Node.contract(tmp, ["phyu"], psi_new[-1], ["d"])]
side[0].rename_leg({"l":"up"})
for i in range(L-2, 0, -1):
    tmp = Node.contract(operator[i], ["phyd"], psi_new, ["phy"])
    tmp.rename_leg({"l":"down", "ol":"mid"})
    tmp = Node.contract(tmp, ["r", "or"], side[0], ["down", "mid"])
    tmp = Node.contract(tmp, ["phyu", "up"], psi_new[i], ["phy", "r"])
    tmp.rename_leg({"l":"up"})
    side = [tmp] + side
side = [Node(["up", "mid", "down"], [D, D, D])] + side

##main part
for _ in range(100):
    #from left to right
    for i in range(L-1):
        tmp = Node.contract(psi0[i], ["phy"], operator[i], ["phyd"])
        tmp.rename_leg({"phyu":"phy"})
        if i is not 0 :
            tmp = Node.contract(tmp, ["l", "ol"], side[i-1], ["mid", "down"])
            tmp.rename_leg({"up":"l"})
        side[i] = Node.copy(tmp)
        tmp = Node(tmp, ["or", "r"], side[i+1], ["mid", "down"])
        tmp.rename_leg({"up":"r"})
        psi_new[i].data , r = Node.qr(tmp, ["r"])
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
            tmp = Node.contract(tmp, ["r", "or"], side[i+1], ["mid", "down"])
            tmp.rename_leg({"up":"r"})
        side[i] = Node.copy(tmp)
        tmp = Node(tmp, ["ol", "l"], side[i-1], ["mid", "down"])
        tmp.rename_leg({"up":"l"})
        psi_new[i].data , r = Node.qr(tmp, ["l"])
        if i is not 0:
            side[i] = Node.contract(side[i], ["phy", "r"], psi_new[i], ["phy", "r"])
        else:
            side[i] = Node.contract(side[i], ["phy"], psi_new[i], ["phy"])
        side[i].rename_leg({"r":"up"})

##compare ans
