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

    def copy(self):
        T = Node(self.tags, self.dl)
        T.data = self.data
        T.env = self.env
        return T

    def absorbEnv(self,i0,n=1):
        tmp = np.ones(self.dll,dtype=int)
        tmp[i0] = self.dl[i0]
        en = np.power(self.env[i0], n)
        self.data *= np.reshape(en,tmp)

    def absorbAllEnv(self,n=1):
        for i in range(self.dll):
            self.absorbEnv(i,n)

    @staticmethod
    def contract(T1,tags1,T2,tags2):
        ##contract 2 tensors
        #order:the indexs of legs waiting for contracting
        order1 = [T1.tags.index(i) for i in tags1]
        order2 = [T2.tags.index(i) for i in tags2]
        #absorb environment
        for i in order1:
            T1.absorbEnv(i)
        for i in order2:
            T2.absorbEnv(i)
        #generate the contribute of the answer
        tags = []
        dl = []
        env = []
        for i in range(T1.dll):
            if not (i in order1):
                tags.append(T1.tags[i])
                dl.append(T1.dl[i])
                env.append(T1.env[i])
        for i in range(T2.dll):
            if not (i in order2):
                tags.append(T2.tags[i])
                dl.append(T2.dl[i])
                env.append(T2.env[i])
        #initiate the answer
        T = Node(tags,dl)
        T.data = np.tensordot(T1.data,T2.data,[order1,order2])
        T.env = env;
        #pollish the repeated tags
        for x in T1.tags:
            if x in tags1:
                continue
            if (not (x in tags2)) & (x in T2.tags):
                T.tags[T.tags.index(x,T.tags.index(x)+1)] += "'"
        return T

    @staticmethod
    def update(T1,tag1,T2,tag2,H):
        ##  SVD
        #0 初始缩并
        i1 = T1.find_leg_index(tag1)
        i2 = T2.find_leg_index(tag2)
        TempT = Node.contract(T1,[tag1],T2,[tag2])
        TempT = Node.contract(TempT,["phy","phy'"],H,["lowerLeft","lowerRight"])
        #1 乘 Env
        TD = TempT.copy()
        TD.absorbAllEnv(2)
        TD = TD.data
        #2 SVD
        temp = list(range(T1.dll-2)) + [TempT.dll-2] + list(range(T1.dll-2,TempT.dll-2)) + [TempT.dll-1]
        TD = np.transpose(TD,temp)
        sh = TD.shape
        sh1 = list(sh[:T1.dll-1])
        sh2 = list(sh[T1.dll-1:])
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
        o1 = list(range(i1)) + [-1] + list(range(i1,T1.dll-1))
        o2 = list(range(1,i2+1)) + [0] + list(range(i2+1,T2.dll))
        T1.data = np.transpose(U,o1)
        T2.data = np.transpose(V,o2)
        #3 吐Env
        T1.absorbAllEnv(-2)
        T1.absorbEnv(i1,2)
        T2.absorbAllEnv(-2)
        T2.absorbEnv(i2,2)
        T1.data/=np.max(np.abs(T1.data))
        T2.data/=np.max(np.abs(T2.data))
