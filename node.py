# -*- coding: utf-8 -*-

import numpy as np
from functools import reduce

class Node:

    def __init__(self,tags,dims,data=None,envs=None):
        assert len(tags) == len(dims)
        if data is not None:
            self.data = np.reshape(np.array(data,dtype=np.float32),dims)
            self.data /= np.max(np.abs(self.data))
        else:
            self.data = np.random.random(dims)
        if envs is not None:
            self.envs = []
            for i,j in zip(dims,envs):
                if j is None:
                    self.envs.push(np.ones(i))
                else:
                    tmp = np.array(j,dtype=np.float32)
                    tmp = tmp/np.max(np.abs(tmp))
                    assert tmp.shape == i
                    self.envs.push(tmp)
        else:
            self.envs = [np.ones(i) for i in dims]
        self.dims = np.array(dims)
        self.tags = np.array(tags)

    def replace(self,other):
        self.data = other.data
        self.envs = other.envs
        self.dims = other.dims
        self.tags = other.tags

    @staticmethod
    def copy(self):
        return Node(self.tags,self.dims,self.data,self.envs)

    def rename_leg(self,tag_dict):
        for i,j in tag_dict.items():
            self.tags[self.find_leg_index(i)] = j

    @staticmethod
    def absorb_envs(self,pow,legs=None):
        if legs == None:
            legs = range(len(self.dims))
        for i in legs:
            tmp = np.ones(len(self.dims),dtype=int)
            tmp[i] = sself.dims[i]
            ans *= np.reshape(np.power(self.envs[i],pow),tmp)
        return ans

    def transpose(self,tags):
        self.data = np.transpose(self.data,[self.tags.index(i) for i in tags])
        tmp = self.dims
        self.dims = [tmp[self.tags.index(i)] for i in tags]
        tmp = self.envs
        self.envs = [tmp[self.tags.index(i)] for i in tags]
        self.tags = tags

    @staticmethod
    def transpose(self,tags):
        data = np.transpose(self.data,[self.tags.index(i) for i in tags])
        dims = [self.dims[self.tags.index(i)] for i in tags]
        envs = [self.envs[self.tags.index(i)] for i in tags]
        return Node(tags,dims,data,envs)

    def __repr__(self):
        return "Node with dims: %s"%str(self.tags)

    @staticmethod
    def contract(T1,tags1,T2,tags2):
        # order:the indexs of legs waiting for contracting
        order1 = [T1.tags.index(i) for i in tags1]
        order2 = [T2.tags.index(i) for i in tags2]
        # absorb envsironment
        TD1 = Node.absorb_envs(T1,1,order1)
        TD2 = Node.absorb_envs(T2,1,order2)
        # generate the contribute of the answer
        tags = [j for i,j in enumerate(T1.tags) if i not in order1] + [j for i,j in enumerate(T2.tags) if i not in order2]
        dims = [j for i,j in enumerate(T1.dims) if i not in order1] + [j for i,j in enumerate(T2.dims) if i not in order2]
        envs = [j for i,j in enumerate(T1.envs) if i not in order1] + [j for i,j in enumerate(T2.envs) if i not in order2]
        assert len(tags) == len(set(tags))
        #initiate the answer
        T = Node(tags,dims,np.tensordot(TD1,TD2,[order1,order2],envs))
        return T

    @staticmethod
    def svd(self,num,tag1,tag2,cut):
        dims1 = self.dims[:num]
        dims2 = self.dims[num:]
        data1, env, data2 = np.linalg.svd(
            np.reshape(
                Node.absorb_envs(self,2)
                [np.prod(dims1),np.prod(dims2)])
        )
        tags1 = self.tags[:num] + [tag1]
        tags2 = [tag2] + self.tags[num:]
        dims1 = dims1 + [cut]
        dims2 = [cut] + dims2
        envs1 = self.envs[:num] + [env]
        envs2 = [env] + self.envs[num:]
        T1,T2 = Node(tags1,dims1,data1,envs1),Node(tags2,dims2,data2,envs2)
        T1.data = Node.absorb_envs(T1,-2,range(len(dims1)-1))
        T2.data = Node.absorb_envs(T2,-2,range(1,len(dims2)))
        return T1,T2

    @staticmethod
    def update(T1,tag1,T2,tag2,phy_leg,H,cut=None):
        # 准备
        l1 = T1.dims[T1.tags.index(phy_leg[0])]
        l2 = T2.dims[T2.tags.index(phy_leg[1])]
        assert T1.dims[T1.tags.index(tag1)] = T2.dims[T2.tags.index(tag2)]
        if cut is None:
            cut = T1.dims[T1.tags.index(tag1)]

        # 缩并
        TD = Node.contract(T1,[tag1],T2,[tag2])
        tmp = TD.tags
        HH = Node(["__1","__2"]+phy_leg,[l1,l2,l1,l2],H)
        TD = Node.contract(TD,phy_leg,H,["__1","__2"]
        TD.transpose(tmp)
        # SVD
        TD1,TD2 = Node.svd(TD,len(T1.tags)-1,tag1,tag2,cut)
        TD1.transpose(T1.tags)
        TD2.transpose(T2.tags)
        T1.replace(TD1)
        T2.replace(TD2)
