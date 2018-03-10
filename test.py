import numpy as np
from square_lattice import *

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

for func in [SimpleNode.qr_update, SimpleNode.update]:
    print("\n",func,"\n")
    A = SimpleNode(["r","l","p"],[2,2,2],[2,3,5,3,7,9,2,1],
                    envs=[[2,1],[10,3],None])
    B = SimpleNode(["r","l","p"],[2,2,2],[3,2,1,6,6,8,3,1],
                    envs=[[3,5],[10,3],None])
    H = np.reshape([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],[2,2,2,2])
    func(A,B,"l","l","p","p",H)
    A.normize()
    B.normize()
    print(A.data)
    print(A.envs)
    break
