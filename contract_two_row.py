import numpy as np
from node import Node

L = 20
d = 2
D1 = 4
D2 = 8

def decompose_tool(func, T, tag, tag1, tag2):
    tmp_tag = T.tags
    tmp = tmp_tag.index(tag)
    T.transpose(tmp_tag[:tmp] + tmp_tag[tmp+1:] + [tag])
    q, r = func(T, len(T.tags)-1, tag1, tag2)
    T.transpose(tmp_tag)
    return q, r

#from right to leftt
#contract two rows into one row

##generate psi0(left, up, right)
psi0 = [Node(["phy", "r"], [d, D1], normf=False)] \
     + [Node(["l", "phy", "r"], [D1, d, D1], normf=False) for _ in range(L-2)] \
     + [Node(["l", "phy"], [D1, d], normf=False)]

##generate Operator(left, up, down, right)
operator = [Node(["phyu", "phyd", "or"], [d, d, D1])] + [Node(["ol", "phyu", "phyd", "or"], [D1, d, d, D1]) for _ in range(L-2)] + [Node(["ol", "phyu", "phyd"], [D1, d, d])]

##psi_new = psi0
psi_new = [Node(["phy", "r"], [d, D2], normf=False)] \
        + [Node(["l", "phy", "r"], [D2, d, D2], normf=False) for _ in range(L-2)] \
        + [Node(["l", "phy"], [D2, d], normf=False)]

##right unitarinalize
for i in range(L-1, 0, -1):
    psi_new[i] , r = decompose_tool(Node.qr, psi_new[i], "l", "l", "r")
    psi_new[i-1] = Node.contract(psi_new[i-1], ["r"], r, ["l"])

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

##compare ans
for i in range(L):
    if i==0:
        ans1 = Node.contract(psi_new[i], ["phy"], psi_new[i], ["phy"], {"r":"up"}, {"r":"down"})
    else:
        ans1 = Node.contract(ans1, ["up"], psi_new[i], ["l"])
        if i is not L-1:
            ans1.rename_leg({"r":"up"})
        ans1 = Node.contract(ans1, ["down", "phy"], psi_new[i], ["l", "phy"])
        if i is not L-1:
            ans1.rename_leg({"r":"down"})
print(ans1.data)

for i in range(L):
    tmp = Node.contract(psi0[i], ["phy"], operator[i], ["phyd"])
    if i==0:
        ans2 = Node.contract(tmp, ["phyu"], tmp, ["phyu"], {"r":"1", "or":"2"}, {"r":"4","or":"3"})
    elif i==L-1:
        ans2 = Node.contract(ans2, ["1", "2"], tmp, ["l", "ol"])
        ans2 = Node.contract(ans2, ["3", "4", "phyu"], tmp, ["ol", "l", "phyu"])
    else:
        ans2 = Node.contract(ans2, ["1", "2"], tmp, ["l", "ol"], {}, {"r":"1", "or":"2"})
        ans2 = Node.contract(ans2, ["3", "4", "phyu"], tmp, ["ol", "l", "phyu"], {}, {"r":"4", "or":"3"})
print(ans2.data)
