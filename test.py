# -*- coding: utf-8 -*-

import numpy as np
from node import Node

A = Node(["l"],[2],2,data=[[1,10],[2,20]],env=[[1,3]])
B = Node(["l"],[2],2,data=[[5,10],[7,20]],env=[[1,3]])

Node.connect(A,"l",B,"l")

H = np.reshape([1,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,1],[2,2,2,2])

Node.update(A,"l",B,"l",H)
Node.update(A,"l",B,"l",H)
Node.update(A,"l",B,"l",H)

