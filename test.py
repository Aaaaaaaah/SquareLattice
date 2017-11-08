# -*- coding: utf-8 -*-

import numpy as np
from node import Node

A = Node(["l"],[2],2,data=[[1,10],[2,20]],env=[[1,2]])
B = Node(["l"],[2],2,data=[[3,300],[4,400]],env=[[1,2]])

Node.connect(A,"l",B,"l")

H = np.reshape(np.identity(4),[2,2,2,2])

Node.update(A,"l",B,"l",H)

print A.data
print B.data
print A.env
print B.env

Node.update(A,"l",B,"l",H)

print A.data
print B.data
print A.env
print B.env
