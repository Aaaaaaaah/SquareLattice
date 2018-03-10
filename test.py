import numpy as np
from square_lattice import *

for func in [SimpleNode.qr_update, SimpleNode.update]:
    print("\n",func,"\n")
    A = SimpleNode(["l","p"],[2,2],[[1,2],[10,30]])
    B = SimpleNode(["l","p"],[2,2],[[5,30],[40,2]])
    H = np.reshape([1,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,1],[2,2,2,2])
    func(A,B,"l","l","p","p",H)
    print(A.data)
    print(A.envs)
