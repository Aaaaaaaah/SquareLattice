import unittest
import numpy as np
from SquareLattice import *

class TestNode(unittest.TestCase):
    def test_init(self):
        print("\nTest Init\n")
        A = Node(["l","p"],[2,2])
        print(A)
        print(A.data)

    def test_contract(self):
        print("\nTest Contract\n")
        A = Node(["l","p"],[2,2],[[1,10],[2,20]])
        B = Node(["l","p"],[2,2],[[5,10],[7,20]])

        ans = Node.contract(A,["l"],B,["p"],{"p":"pp"},{"l":"ll"})

        print(ans)
        print(ans.data)
        ans = Node.contract(A,["l"],B,["p"])

    def test_rename_tags(self):
        print("\nTest Rename Leg\n")
        A = Node(["l","p"],[2,3])
        print(A)
        A.rename_leg({"p":"r"})
        print(A)
        A.rename_leg(["A","B"])
        print(A)

    def test_svd(self):
        print("\nTest SVD\n")
        A = Node(["l","p"],[2,2],[[1,10],[2,30]])
        B = Node.svd(A,["p"],"l","p")
        print(B[0].data)
        print(B[1])
        print(B[2].data)
        C = Node.svd(A,1,"p","l",cut=1)

    def test_qr(self):
        print("\nTest QR\n")
        A = Node(["l","p"],[2,2],[[1,10],[2,30]])
        B = Node.qr(A,1,"p","l")
        print(B[0].data)
        print(B[1].data)
        C = Node.qr(A,["p"],"l","p",cut=1)

    def test_reshape(self):
        print("\nTest Reshape\n")
        A = Node(["l","a","b"],[2,2,2])
        A.reshape(["l","r"],[2,4])
        print(A)

class TestSimpleNode(unittest.TestCase):
    def test_init(self):
        print("\nTest Init\n")
        A = SimpleNode(["a","b"],[2,2],[[1,10],[2,30]],envs=[[2,0.1],None])
        print(A)
        print(A.envs)
        B = SimpleNode(["a","b"],[2,2],[[1,10],[2,30]])
        print(B.envs)
        C = SimpleNode(["a","b"],[2,2],[[1,10],[2,30]],init_envs=False)
        print(C.envs)

    def test_contract(self):
        print("\nTest Contract\n")
        A = SimpleNode(["l","p"],[2,2],[[1,10],[2,20]])
        B = SimpleNode(["l","p"],[2,2],[[5,10],[7,20]])
        SimpleNode.connect(A,"l",B,"p")
        A.envs[0][0] = 0.5
        A.envs[0][1] = 0.5

        ans = SimpleNode.contract(A,["l"],B,["p"],{"p":"pp"},{"l":"ll"})

        print(ans)
        print(ans.data)
        print(B.envs[1])
        ans = SimpleNode.contract(A,["l"],B,["p"])

    def test_transpose(self):
        print("\nTest Transpose\n")
        A = SimpleNode(["l","p"],[2,2],[[1,10],[2,20]])
        B = SimpleNode.transpose(A,["p","l"])
        print(A.data)
        print(B.data)

    def test_svd(self):
        print("\nTest SVD\n")#####???????
        A = SimpleNode(["l","p"],[2,2],[[1,10],[2,30]],envs=[[2,1],[1,2]])
        B = SimpleNode.svd(A,["p"],"l","p")
        print(B[2].data)
        print(B[0].data)
        print(B[1].data)
        print(B[0].envs)
        print(B[1].envs)
        C = SimpleNode.svd(A,1,"p","l",cut=1)

    def test_absorb(self):
        print("\nTest absorb\n")
        A = SimpleNode(["l","p"],[2,2],[[1,10],[2,30]],envs=[[0.5,0.1],[0.1,0.5]])
        B = SimpleNode.absorb(A)
        print(B.data)
        print(type(B))

    def test_update(self):
        print("\nTest Update\n")
        A = SimpleNode(["l","p"],[2,2])
        B = SimpleNode(["l","p"],[2,2])
        print(A.data)
        H = np.reshape([1,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,1],[2,2,2,2])
        SimpleNode.update(A,B,"l","l","p","p",H)
        print(A.data)
        print(A.envs)

if __name__ == '__main__':
    unittest.main()
