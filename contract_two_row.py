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
Operator = [Node(["phyu", "phyd", "or"], [d, d, D]) + [Node(["ol", "phyu", "phyd", "or"], [D, d, d, D]) for _ in range(L-2)] + [Node(["ol", "phyu", "phyd"], [D, d, d])]

##psi_new = psi0
psi_new = [Node.copy(i) for i in psi0]

##right unitarinalize
for i in range(L-1, 0, -1):
    r = psi_new[i].qr(["l"])
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
side = [None] + side

##main part
for _ in range(100):
    #from left to right
    for i in range(L-1):
        pass
    #from right to left
    for i in range(L-1, 0, -1):
        pass

##compare ans
