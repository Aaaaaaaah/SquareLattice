# -*- coding: utf-8 -*-

import numpy as np
from functools import reduce

class Node:

    def __init__(self,tags,dl,dp,data=None,env=None):
        if data:
            self.data = np.array(data,dtype=np.float32)
        else:
            self.data = np.random.random(dl+[dp])
        if env:
            self.env = [np.array(i,dtype=np.float32) for i in env ]
        else:
            self.env = [np.ones(i) for i in dl]
        self.dl = dl # dimensions of lattice
        self.dll = len(dl) # length of dimensions of lattice
        self.dp = dp # dimensions of physics
        self.tags = tags # dimensions

    def find_leg_index(self,tag):
        return self.tags.index(tag)

    @staticmethod
    def connect(T1,tag1,T2,tag2):
        i1 = T1.find_leg_index(tag1)
        i2 = T2.find_leg_index(tag2)
        assert T1.dl[i1] == T2.dl[i2]
        T1.env[i1] = T2.env[i2]

    @staticmethod
    def update(T1,tag1,T2,tag2,H):
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
        sh2 = list(sh[T2.dll:])
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
        o2 = list(range(1,i2+1)) + [0] + list(range(i2+1,T1.dll+1))
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
