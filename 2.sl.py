import numpy as np
from square_lattice import *

L1 = 4
L2 = 4
D = 8

#   u
# l   r
#   d

def _node(s):
    l = [D]*len(s)
    l.append(2)
    return SimpleNode(list(s)+["p"],l)

def node_in_lattice(i,j):
    s = "udrl"
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


H = np.reshape([[0.25,0,0,0],[0,-0.25,0.5,0],[0,0.5,-0.25,0],[0,0,0,0.25]],[2,2,2,2])
I = np.reshape(np.identity(4),[2,2,2,2])

ep = 0.1
expH = I - 4.*ep*H

for t in range(30):
    print(t)
    if t == 100:
        ep = 0.05
        expH = I - 4.*ep*H
    elif t == 200:
        ep = 0.01
        expH = I - 4.*ep*H
    elif t == 400:
        ep = 0.001
        expH = I - 4.*ep*H
    elif t == 800:
        ep = 0.0005
        expH = I - 4.*ep*H
    elif t == 1300:
        ep = 0.0001
        expH = I - 4.*ep*H
    for i in range(0,L1):
        for j in range(0,L2-1,2):
            SimpleNode.update(lattice[i][j],lattice[i][j+1],"r","l","p","p",expH,cut=D)
    for i in range(0,L1):
        for j in range(1,L2-1,2):
            SimpleNode.update(lattice[i][j],lattice[i][j+1],"r","l","p","p",expH,cut=D)
    for j in range(0,L2):
        for i in range(0,L1-1,2):
            SimpleNode.update(lattice[i][j],lattice[i+1][j],"d","u","p","p",expH,cut=D)
    for j in range(0,L2):
        for i in range(1,L1-1,2):
            SimpleNode.update(lattice[i][j],lattice[i+1][j],"d","u","p","p",expH,cut=D)
    for i in range(0,L1):
        for j in range(0,L2):
            lattice[i][j].normize()

total = SimpleNode.copy_shape(lattice[0][0])
total.data = lattice[0][0].data
total.envs = lattice[0][0].envs
total.rename_leg({"d":"d0","p":"p0,0"})

for i in range(L1):
    for j in range(L2):
        if i==0 and j==0:
            continue
        if i==0:
            total = SimpleNode.contract(total, ["r"], lattice[i][j], ["l"], {}, {"d":"d%d"%j,"p":"p0,%d"%j})
        elif i!= L1-1:
            if j!=0:
                total = SimpleNode.contract(total, ["r", "d%d"%j], lattice[i][j], ["l", "u"], {}, {"d":"d%d"%j,"p":"p%d,%d"%(i,j)})
            else:
                total = SimpleNode.contract(total, ["d%d"%j], lattice[i][j], ["u"], {}, {"d":"d%d"%j,"p":"p%d,%d"%(i,j)})
        else:
            if j!=0:
                total = SimpleNode.contract(total, ["r", "d%d"%j], lattice[i][j], ["l", "u"], {}, {"p":"p%d,%d"%(i,j)})
            else:
                total = SimpleNode.contract(total, ["d%d"%j], lattice[i][j], ["u"], {}, {"p":"p%d,%d"%(i,j)})

print(total)

HH = SimpleNode(["1","2","3","4"],[2,2,2,2],H)


s = 0

ii = SimpleNode.contract(total,total.tags,total,total.tags).data

tags = set(total.tags)

def cal_energy1(i,j):
    a = "p%d,%d"%(i,j)
    b = "p%d,%d"%(i,j+1)
    lin = list(tags - set([a,b]))
    tmp = SimpleNode.contract(total,lin,total,lin,{a:"1",b:"2"})
    ans = SimpleNode.contract(tmp,[0,1,2,3],HH,[0,1,2,3])
    return ans

for i in range(L1):
    for j in range(L2-1):
        s += cal_energy1(i,j).data

def cal_energy2(i,j):
    a = "p%d,%d"%(i+1,j)
    b = "p%d,%d"%(i,j)
    lin = list(tags - set([a,b]))
    tmp = SimpleNode.contract(total,lin,total,lin,{a:"1",b:"2"})
    ans = SimpleNode.contract(tmp,[0,1,2,3],HH,[0,1,2,3])
    return ans

for i in range(L1-1):
    for j in range(L2):
        s += cal_energy2(i,j).data

print(s/ii/L1/L2)

