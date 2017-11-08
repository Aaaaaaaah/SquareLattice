# -*- coding: utf-8 -*-

import numpy as np
from node import Node

A = Node(["l"],[2],2,data=[[1,10],[2,20]],env=[[1,4]])
B = A.copy()

Node.connect(A,"l",B,"l")

H = np.reshape(np.identity(4),[2,2,2,2])


Node.update(A,"l",B,"l",H)
Node.update(A,"l",B,"l",H)
Node.update(A,"l",B,"l",H)

