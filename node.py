# -*- coding: utf-8 -*-

import numpy as np
from functools import reduce

class Node:

    def __init__(self,tags,dims,data=None,env=None):
        if data is not None:
            self.data = np.array(data,dtype=np.float32)
            self.data /= np.max(np.abs(self.data))
            assert self.data.shape == dims
        else:
            self.data = np.random.random(dims)
        if env is not None:
            self.env = []
            for i,j in zip(env,dims):
                tmp = np.array(i,dtype=np.float32)
                tmp = tmp/np.max(np.abs(tmp))
                assert tmp.shape == j
                self.env.push(tmp)
        else:
            self.env = [np.ones(i) for i in dims]
        self.dims = dims # dimensions of lattice
        self.diml = len(dims) # length of dimensions of lattice
        self.tags = tags # dimensions

    @staticmethod
    def copy(self):
        return Node(self.tags,self.dims,data=self.data,env=self.env)

    def find_leg_index(self,tag):
        return self.tags.index(tag)

    def rename_leg(self,tag_dict):
        for i,j in tag_dict.items():
            self.tags[self.find_leg_index(i)] = j

    @staticmethod
    def absorb_env(self,pow):
        ans = np.copy(self.data)
        for i,j in enumerate(self.env):
            tmp = np.ones(self.diml,dtype=int)
            tmp[i] = self.dims[i]
            ans *= np.reshape(np.power(self.env[i],pow[i]),tmp)
        return ans

    def absorb_env(self,pow)
        for i,j in enumerate(self.env):
            tmp = np.ones(self.diml,dtype=int)
            tmp[i] = self.dims[i]
            self.data *= np.reshape(np.power(self.env[i],pow[i]),tmp)

    def transpose(self,tags):
        self.data = np.transpose(self.data,[self.tags.index(i) for i in tags])
        tmp = self.dims
        self.dims = [tmp[self.tags.index(i)] for i in tags]
        tmp = self.env
        self.env = [tmp[self.tags.index(i)] for i in tags]
        self.tags = tags

    @staticmethod
    def transpose(self,tags):
        data = np.transpose(self.data,[self.tags.index(i) for i in tags])
        dims = [self.dims[self.tags.index(i)] for i in tags]
        env = [self.env[self.tags.index(i)] for i in tags]
        return Node(tags,dims,data,env)

    def __repr__(self):
        return "Node with dims: %s"%str(self.tags)

    @staticmethod
    def contract(T1,tags1,T2,tags2,include_env=False):
        # contract 2 tensors
        # order:the indexs of legs waiting for contracting
        order1 = [T1.tags.index(i) for i in tags1]
        order2 = [T2.tags.index(i) for i in tags2]
        # absorb environment
        if include_env:
            env_order = 2
        else:
            env_order = 0
        TD1 = Node.absorb_env(T1,[1 if i in order1 else env_order for i in range(T1.diml)])
        TD2 = Node.absorb_env(T2,[1 if i in order2 else env_order for i in range(T2.diml)])
        # generate the contribute of the answer
        tags = [j for i,j in enumerate(T1.tags) if i not in order1] + [j for i,j in enumerate(T2.tags) if i not in order2]
        dims = [j for i,j in enumerate(T1.dims) if i not in order1] + [j for i,j in enumerate(T2.dims) if i not in order2]
        env = [j for i,j in enumerate(T1.env) if i not in order1] + [j for i,j in enumerate(T2.env) if i not in order2]
        assert len(tags) == len(set(tags))
        #initiate the answer
        T = Node(tags,dims,np.tensordot(TD1,TD2,[order1,order2],env))
        return T

    @staticmethod
    def svd(self,tags1,tags2,tag1,tag2):
        pass

    @staticmethod
    def update(T1,tag1,T2,tag2,phy_leg,H,cut=None):
        ##  SVD
        # 缩并
        i1 = T1.find_leg_index(tag1)
        l1 = T1.dims[T1.tags.index(phy_leg[0])]
        t1 = T1.tags
        i2 = T2.find_leg_index(tag2)
        l2 = T2.dims[T2.tags.index(phy_leg[1])]
        t2 = T2.tags
        TD = Node.contract(T1,[tag1],T2,[tag2],True)
        tmp = TD.tags
        HH = Node(["__1","__2"]+phy_leg,[l1,l2,l1,l2],H)
        TD = Node.contract(TD,phy_leg,H,["__1","__2"]
        TD.transpose(tmp)
        assert T1.dim[i1] == T2.dim[i2]
        temp = list(range(T1.dll-2)) + [TempT.dll-2] + list(range(T1.dll-2,TempT.dll-2)) + [TempT.dll-1]
        TD = np.transpose(TD,temp)
        # SVD
        Node.svd(TD,cut,)
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
        #<<<<<<< HEAD
        #3 吐Env
        T1.absorbAllEnv(-2)
        T1.absorbEnv(i1,2)
        T2.absorbAllEnv(-2)
        T2.absorbEnv(i2,2)
        #=======
        #5 吐Env
        for i,j in enumerate(T1.env):
            if i is i1:
                continue
            tmp = np.ones(T1.dll+1,dtype=np.int)
            tmp[i] = T1.dl[i]
            T1.data /= np.reshape(j*j,tmp)
        for i,j in enumerate(T2.env):
            if i is i2:
                continue
            tmp = np.ones(T2.dll+1,dtype=np.int)
            tmp[i] = T2.dl[i]
            T2.data /= np.reshape(j*j,tmp)
        #>>>>>>> HaoZhang
        T1.data/=np.max(np.abs(T1.data))
        T2.data/=np.max(np.abs(T2.data))
