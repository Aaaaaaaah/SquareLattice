# -*- coding: utf-8 -*-

import numpy as np
from node import Node

A = Node(["l","p"],[2,2],[[1,10],[2,20]],[[1,3],None])
B = Node(["l","p"],[2,2],[[5,10],[7,20]],[[1,3],None])

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
