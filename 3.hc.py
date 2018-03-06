import numpy as np
from square_lattice import *

L1 = 5
L2 = 4
D = 4

#   u
# l   r
#   d

# +-+ +-+ +-+
# | | | | | |
# + +-+ +-+ +
# | | | | | |
# +-+ +-+ +-+
# | | | | | |
# + +-+ +-+ +
# | | | | | |
# +-+ +-+ +-+

def _node(s):
    l = [D]*len(s)
    l.append(2)
    return SimpleNode(list(s)+["p"],l)

def node_in_lattice(i,j):
    if (i-j)%2 == 0:
        s = "udr"
    else:
        s = "udl"
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

ep = 0.1

H = np.reshape([[0.25,0,0,0],[0,-0.25,0.5,0],[0,0.5,-0.25,0],[0,0,0,0.25]],[2,2,2,2])
I = np.reshape(np.identity(4),[2,2,2,2])
expH = lambda x:I - x*ep*H

for t in range(200):
    print(t)
    if(t==50):
        ep = 0.01
    for i in range(0,L1):
        for j in range(i%2,L2-1,2):
            SimpleNode.update(lattice[i][j],lattice[i][j+1],"r","l","p","p",expH(1),cut=D)
    for j in range(0,L2):
        for i in range(j%2,L1-1,2):
            SimpleNode.update(lattice[i][j],lattice[i+1][j],"d","u","p","p",expH(2),cut=D)
    for j in range(0,L2):
        for i in range((j+1)%2,L1-1,2):
            SimpleNode.update(lattice[i][j],lattice[i+1][j],"d","u","p","p",expH(3),cut=D)
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
            if "r" in total.tags:
                total = SimpleNode.contract(total, ["r"], lattice[i][j], ["l"], {}, {"d":"d%d"%j,"p":"p0,%d"%j})
            else:
                total = SimpleNode.contract(total, [], lattice[i][j], [], {}, {"d":"d%d"%j,"p":"p0,%d"%j})
        elif i!= L1-1:
            if j!=0:
                if "r" in total.tags:
                    total = SimpleNode.contract(total, ["r", "d%d"%j], lattice[i][j], ["l", "u"], {}, {"d":"d%d"%j,"p":"p%d,%d"%(i,j)})
                else:
                    total = SimpleNode.contract(total, ["d%d"%j], lattice[i][j], ["u"], {}, {"d":"d%d"%j,"p":"p%d,%d"%(i,j)})
            else:
                total = SimpleNode.contract(total, ["d%d"%j], lattice[i][j], ["u"], {}, {"d":"d%d"%j,"p":"p%d,%d"%(i,j)})
        else:
            if j!=0:
                if "r" in total.tags:
                    total = SimpleNode.contract(total, ["r", "d%d"%j], lattice[i][j], ["l", "u"], {}, {"p":"p%d,%d"%(i,j)})
                else:
                    total = SimpleNode.contract(total, ["d%d"%j], lattice[i][j], ["u"], {}, {"p":"p%d,%d"%(i,j)})
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

