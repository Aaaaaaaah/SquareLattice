import numpy as np
from node import Node
import sl

tmp = Node([], [], data = [1], normf=False)
for i,x in enumerate(sl.lattice):
    for j,y in enumerate(x):
        for k in ["u","d","r","l","p"]:
            if k in y.tags:
                y.rename_leg({k:k+str(i)+str(j)})

for i,j in enumerate(sl.lattice):
    for k,p in enumerate(j):
        tags1 = []
        tags2 = []
        if (i-1) in range(4):
            tags1.append("d%s%s"%(str(i-1),str(k)))
            tags2.append("u%s%s"%(str(i),str(k)))
        if (k-1) in range(4):
            tags1.append("r%s%s"%(str(i),str(k-1)))
            tags2.append("l%s%s"%(str(i),str(k)))
        tmp = Node.contract(tmp, tags1, p, tags2)

norm = Node.contract(tmp, ["p%s%s" % (str(i),str(j)) for j in range(4) for i in range(4)] \
                     , tmp, ["p%s%s" % (str(i),str(j)) for j in range(4) for i in range(4)])
print("norm = ", norm.data)

H = Node(["d1","d2","u1","u2"], [2,2,2,2], np.array(sl.H), normf=False)
E = np.array([0.0])
for i in range(4):
    for j in range(3):
        tmp2 = Node.contract(tmp, ["p%s%s"%(str(i),str(j)), "p%s%s"%(str(i),str(j+1))], H, ["d1","d2"], {}, \
                             {"u1":"p%s%s"%(str(i),str(j)),"u2":"p%s%s"%(str(i),str(j+1))})
        tmp2 = Node.contract(tmp2, ["p%s%s" % (str(p),str(q)) for p in range(4) for q in range(4)], tmp, \
                             ["p%s%s" % (str(p),str(q)) for p in range(4) for q in range(4)])
        E += tmp2.data / norm.data
        tmp2 = Node.contract(tmp, ["p%s%s"%(str(j),str(i)), "p%s%s"%(str(j+1),str(i))], H, ["d1","d2"], {}, \
                             {"u1":"p%s%s"%(str(j),str(i)),"u2":"p%s%s"%(str(j+1),str(i))})
        tmp2 = Node.contract(tmp2, ["p%s%s" % (str(p),str(q)) for p in range(4) for q in range(4)], tmp, \
                             ["p%s%s" % (str(p),str(q)) for p in range(4) for q in range(4)])
        E += tmp2.data / norm.data
print(E / 16.0)
