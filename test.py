import unittest
import numpy as np
from sq import *

class TestSquareLattice(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()

"""
H = np.reshape([1,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,1],[2,2,2,2])
Node.update(A,B,"l","l","p","p",H)
print A.data
print B.data
Node.update(A,B,"l","l","p","p",H)
print A.data
print B.data
Node.update(A,B,"l","l","p","p",H)
print A.data
print B.data
"""

"""
Correct output
[[-0.1  1. ]
[ 1.   0.1 ]]
[[-0.35405405  1.        ]
[ 1.          0.35405405 ]]
[[-0.1 -1. ]
[-1.   0.1]]
[[-0.35405405 -1.        ]
[ 1.         -0.35405405 ]]
[[-0.1  1. ]
[ 1.   0.1 ]]
[[-0.35405405  1.        ]
[-1.         -0.35405405]]
"""
