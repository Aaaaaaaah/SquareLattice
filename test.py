# -*- coding: utf-8 -*-

import numpy as np
from node import Node

A = Node(["l"],[2],2,data=[[1,2],[3,4]],env=[[1,2]])
B = Node(["l"],[2],2,data=[[5,2],[7,3]],env=[[1,2]])

Node.connect(A,"l",B,"l")

#H= np.reshape([[0.25,0,0,0],[0,-0.25,0.5,0],[0,0.5,-0.25,0],[0,0,0,0.25]],[2,2,2,2])

H = np.reshape(np.identity(4),[2,2,2,2])

Node.update(A,"l",B,"l",H)

print A.data
print B.data
print A.env
print B.env
