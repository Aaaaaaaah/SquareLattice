# -*- coding: utf-8 -*-

import numpy as np
from functools import reduce

class Node:

    def __init__(self,tags,dl):
        self.data = np.random.random(dl)
        self.env = [np.ones(i) for i in dl]
        self.dl = dl # dimensions of lattice
        self.dll = len(dl) # length of dimensions of lattice
        self.tags = tags # dimensions

    def find_leg_index(self,tag):
        return self.tags.index(tag)

    def rename(self,oldTag,newTag):
        assert len(oldTag) == len(newTag)
        for i in range(len(oldTag)):
            self.tags[self.find_leg_index(oldTag[i])] = newTag[i]

    def trace(self,tags1,tags2):
        assert len(tags1) == len(tags2)
        assert len(tags1 + tags2) == len(set(tags1 + tags2))
        for i in range(len(tags1)):
            temp = [self.find_leg_index(tags1[i])]
            temp += [self.find_leg_index(tags2[i])]
            i1 = min(temp)
            i2 = max(temp)
            self.data = self.data.trace(0,i1,i2)
            self.dl = self.dl[:i1] + self.dl[i1+1:i2] + self.dl[i2+1:]
            self.dll -= 2

    @staticmethod
    def contract(T1,tags1,T2,tags2):
        dl = T1.dl + T2.dl
        tags = T1.tags + T2.tags
        order1 = [T1.tags.index(i) for i in tags1]
        order2 = [T2.tags.index(i) for i in tags2]
        for i in tags1 + tags2:
            temp = tags.index(i)
            tags = tags[:temp] + tags[temp+1:]
            dl = dl[:temp] + dl[temp+1:]
        T = Node(tags,dl)
        T.data = np.tensordot(T1.data,T2.data,[order1,order2])
        for x in T1.tags:
            if x in tags1:
                continue
            if (not (x in tags2)) & (x in T2.tags):
                T2.tags[T2.tags.index(x)] += "'"
        return T

    #原有的connect由于env初始化为1没什么用了
    @staticmethod
    def connect(T1,tag1,T2,tag2):
        i1 = T1.find_leg_index(tag1)
        i2 = T2.find_leg_index(tag2)
        assert T1.dl[i1] == T2.dl[i2]
        T1.env[i1] = T2.env[i2]

    @staticmethod
    def update(T1,tag1,T2,tag2,H):
        ##1 TempT = contract(T1,tag1,T2,Tag2)
        ##2 TempT = contract(TempT,"phy1",H,"lowerLeft")
        ##3 TempT = contract(TempT,"phy2",H,"lowerRight")
        ##  SVD
        #1 乘 Env
        i1 = T1.find_leg_index(tag1)
        i2 = T2.find_leg_index(tag2)
        assert T1.dl[i1] == T2.dl[i2]
        TD1 = T1.data.copy()
        TD2 = T2.data.copy()
        for i,j in enumerate(T1.env):
            tmp = np.ones(T1.dll+1,dtype=np.int)
            tmp[i] = T1.dl[i]
            TD1 *= np.reshape(j*j,tmp)
        for i,j in enumerate(T2.env):
            tmp = np.ones(T2.dll+1,dtype=np.int)
            tmp[i] = T2.dl[i]
            TD2 *= np.reshape(j*j,tmp)
        #2 两个Tensor相乘
        TD = np.tensordot(TD1,TD2,[[i1],[i2]])
        #3 乘上Hamiltonian
        TD = np.tensordot(TD,H,[[T1.dll-1,-1],[0,1]])
        tmp = list(range(T1.dll-1))+[T1.dll+T2.dll-2]+list(range(T1.dll-1,T1.dll+T2.dll-2))+[T1.dll+T2.dll-1]
        TD = np.transpose(TD,tmp)
        #4 SVD
        sh = TD.shape
        sh1 = list(sh[:T1.dll])
        sh2 = list(sh[T1.dll:])
        TD = np.reshape(TD,[reduce(int.__mul__,sh1),reduce(int.__mul__,sh2)])
        U,S,V = np.linalg.svd(TD)
        T1.env[i1]=np.sqrt(S[:T1.dl[i1]])
        T2.env[i2]=np.sqrt(S[:T2.dl[i2]])
        T1.env[i1]/=np.max(np.abs(T1.env[i1]))
        T2.env[i2]/=np.max(np.abs(T2.env[i2]))
        sh1 = sh1 + [T1.dl[i1]]
        sh2 = [T2.dl[i2]] + sh2
        U = np.reshape(U[:,:T1.dl[i1]],sh1)
        V = np.reshape(V[:T2.dl[i2],:],sh2)
        o1 = list(range(i1)) + [-1] + list(range(i1,T1.dll))
        o2 = list(range(1,i2+1)) + [0] + list(range(i2+1,T2.dll+1))
        T1.data = np.transpose(U,o1)
        T2.data = np.transpose(V,o2)
        #5 吐Env
        for i,j in enumerate(T1.env):
            if i == i1:
                continue
            tmp = np.ones(T1.dll+1,dtype=np.int)
            tmp[i] = T1.dl[i]
            TD1 /= np.reshape(j*j,tmp)
        for i,j in enumerate(T2.env):
            if i == i2:
                continue
            tmp = np.ones(T2.dll+1,dtype=np.int)
            tmp[i] = T2.dl[i]
            TD2 /= np.reshape(j*j,tmp)
        T1.data/=np.max(np.abs(T1.data))
        T2.data/=np.max(np.abs(T2.data))
        print(T1.env[0])
        print(T1.env[1])
