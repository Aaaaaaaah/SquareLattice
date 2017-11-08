# -*- coding: utf-8 -*-

import numpy as np
from node import Node

D = 2

AD = np.array([[[ 0.10018639,  0.99187046 ],
                      [ 0.4107053 ,  0.29783095 ]],
                     [[ 0.40567293,  0.6578474  ],
                             [ 0.6066298 ,  0.91806249 ]]])
BD = np.array([[[ 0.98101047,  0.68772476 ],
                      [ 0.59211107,  0.58222698 ]],
                     [[ 0.40631414,  0.9213826  ],
                             [ 0.86095451,  0.52654028 ]]])

A = Node(["l","r","p"],[D,D,2],AD)
B = Node(["l","r","p"],[D,D,2],BD)

Node.connect(A,"l",B,"r")
Node.connect(A,"r",B,"l")

ep = 0.1

H = np.reshape([[0.25,0,0,0],[0,-0.25,0.5,0],[0,0.5,-0.25,0],[0,0,0,0.25]],[2,2,2,2])
I = np.reshape(np.identity(4),[2,2,2,2])
expH = I - 4.*ep*H

for _ in range(1):
    Node.update(A,B,"r","l","p","p",expH)
    Node.update(A,B,"l","r","p","p",expH)
    Node.update(A,B,"l","r","p","p",expH)
    Node.update(A,B,"r","l","p","p",expH)

print A.envs

"""
Correct output is
[array([ 1.        ,  0.38003505 ]), array([ 1.        ,  0.32616988 ]),
array([ 1.,  1.])]
"""
