# -*- coding: utf-8 -*-

import numpy as np
from node import Node

D = 8

A = Node(["l","r","p"],[D,D,2])
B = Node(["l","r","p"],[D,D,2])

Node.connect(A,"l",B,"r")
Node.connect(A,"r",B,"l")

ep = 0.1

H = np.reshape([[0.25,0,0,0],[0,-0.25,0.5,0],[0,0.5,-0.25,0],[0,0,0,0.25]],[2,2,2,2])
I = np.reshape(np.identity(4),[2,2,2,2])
expH = I - 4.*ep*H

for _ in range(100):
    Node.update(A,B,"r","l","p","p",expH)
    Node.update(A,B,"l","r","p","p",expH)
    Node.update(A,B,"l","r","p","p",expH)
    Node.update(A,B,"r","l","p","p",expH)

print(A.envs[:2])
