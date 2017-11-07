import numpy as np
from node import Node

D = 4
A = Node(["l","r","phy"],[D,D,2])
B = Node(["l","r","phy"],[D,D,2])

#Node.connect(A,"l",B,"r")
#Node.connect(A,"r",B,"l")

ep = 0.1

H = np.reshape([[0.25,0,0,0],[0,-0.25,0.5,0],[0,0.5,-0.25,0],[0,0,0,0.25]],[2,2,2,2])
I = np.reshape(np.identity(4),[2,2,2,2])
expH = Node(["lowerLeft","lowerRight","upperLeft","upperRight"],[2,2,2,2])
expH.data = I - 4.*ep*H

for _ in range(1000):
    Node.update(A,"l",B,"r",expH)
    Node.update(A,"r",B,"l",expH)

Left = Node(["u","d"],[D,D])
Right = Node(["u","d"],[D,D])
for _ in range(1000):
    Left = Node.contract(Left,["u"],A,["l"])
    Left = Node.contract(Left,["phy","d"],A,["phy","l"])
    Left.rename(["r","r'"],["u","d"])
    Left = Node.contract(Left,["u"],B,["l"])
    Left = Node.contract(Left,["phy","d"],B,["phy","l"])
    Left.rename(["r","r'"],["u","d"])
    Right = Node.contract(Right,["u"],B,["r"])
    Right = Node.contract(Right,["phy","d"],B,["phy","r"])
    Right.rename(["l","l'"],["u","d"])
    Right = Node.contract(Right,["u"],A,["r"])
    Right = Node.contract(Right,["phy","d"],A,["phy","r"])
    Right.rename(["l","l'"],["u","d"])

temp = Node.contract(A,["r"],B,["l"])
Norm = Node.contract(Left,["u"],temp,["l"])
Norm = Node.contract(Norm,["d","phy","phy'"],temp,["l","phy","phy'"])
Norm = Node.contract(Norm,["r","r'"],Right,["u","d"])

Energy = Node.contract(Left,["u"],temp,["l"])
temp = Node.contract(temp,["phy","phy'"],expH,["lowerLeft","lowerRight"])
Energy = Node.contract(Energy,["d","phy","phy'"],temp,["l","upperLeft","upperRight"])
Energy = Node.contract(Energy,["r","r'"],Right,["u","d"])

print(Energy.data / Norm.data)
