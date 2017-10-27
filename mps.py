# -*- coding: utf-8 -*-

import numpy as np
from node import Node

A = Node(["l","r","phy"],[5,3,2])
B = Node(["l","r","phy"],[3,5,2])

#Node.connect(A,"l",B,"r")
#Node.connect(A,"r",B,"l")

ep = 0.1

H = np.reshape([[0.25,0,0,0],[0,-0.25,0.5,0],[0,0.5,-0.25,0],[0,0,0,0.25]],[2,2,2,2])
I = np.reshape(np.identity(4),[2,2,2,2])
expH = Node(["lowerLeft","lowerRight","upperRight","upperLeft"],[2,2,2,2])
expH.data = I - 4.*ep*H

for _ in range(10):
    Node.update(A,"l",B,"r",expH)
    Node.update(A,"r",B,"l",expH)
