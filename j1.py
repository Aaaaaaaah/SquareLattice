# -*- coding: utf-8 -*-
"""
This module include class Node
"""

import numpy as np
import tensorflow as tf

import node

class J1Node(node.Node):
    """Node of the lattice

    Represent one node in the whole network and support necessary
    operations on node.

    Attribute:
        tags: the name of each dimension
        dims: the order of each dimension
        data: the tensor data of the node.
        envs: the environments of each dimension
        envf: if it is using environments, envf is True, else False
        normf: if it is using autonormalization, normf is True, else False
    """

    # 初始化函数
    def __init__(self, tags, dims, data=None, envs=None):
        super(J1Node, self).__init__(tags, dims, data)
        if envs is None:
            envs = [None for _ in tags]
        self.envs = []
        for i, j in zip(dims, envs):
            if node.BASE == "NP":
                if j is None:
                    tmp = np.ones(i)
                else:
                    tmp = np.array(j, dtype=np.float32)
                    assert tmp.shape == (i,)
            elif node.BASE == "TF":
                if j is None:
                    tmp = tf.ones(i)
                else:
                    tmp = tf.cast(j, dtype=tf.float32)
                    assert tmp.shape == (i,)
            else:
                raise Exception("FT is TODO")
            self.envs.append(tmp)

    @staticmethod
    def absorb_envs(tensor, pows, legs=None):
        """Absorb environments into data

        Absorb environments into data and return the tensor.
        also release environments function can be obtained by
        pows below 0

        Args:
            tensor: the specific Node object to operate
            pows: the times to absorb the environments, when this
                argument is below 0, it means release.
            legs: determine which dimensions to be absorb.
                if legs are none, then absorb all dimensions.

        Returns:
            ans: the tensor of data having absorbed the environments.
        """
        ans = tensor.data.copy()
        if legs is None:
            legs = range(len(tensor.dims))
        for i in legs:
            tmp = np.ones(len(tensor.dims), dtype=int)
            tmp[i] = tensor.dims[i]
            ans *= np.reshape(np.power(tensor.envs[i], pows), tmp)
        return ans

    @property
    def envf(self):
        return self.__envf

    @property
    def normf(self):
        return self.__normf

    @envf.setter
    def envf(self, value):
        if value is True and self.__envf is False:
            self.data = Node.absorb_envs(self, -1)
            self.__envf = True
        if value is False and self.__envf is True:
            self.data = Node.absorb_envs(self, 1)
            self.envs = [np.ones(self.dims[i]) for i in range(len(self.dims))]
            self.__envf = False

    @normf.setter
    def normf(self, value):
        if value and not self.__normf:
            self.data /= np.max(np.abs(self.data))
        self.__normf = value

    def matrix_multiply(self, tag, r, r_ind=0):
        self.envf = False
        tbak = list(self.tags)
        ind = self.tags.index(tag)
        del self.tags[ind]
        self.tags.append(tag)
        self.data = np.tensordot(self.data, r, ((ind), (r_ind)))
        self.transpose(tbak)

    #张量操作

    # simple update
    @staticmethod
    def update(T1, T2, tag1, tag2, phy1, phy2, H, cut=None):
        """Update two tensor with Hamiltonian

        update in Simple Update method

        Args:
            T1, T2: tensor wait to be update
            phy1, phy2: the physical dimension of T1, T2
            tag1, tag2: the dimension wait to be contracted
            H: Hamiltonian
            cut: the number of singularvalue remain
        """
        # 准备
        l1 = T1.dims[T1.tags.index(phy1)]
        l2 = T2.dims[T2.tags.index(phy2)]
        if cut is None:
            cut = T1.dims[T1.tags.index(tag1)]
        # 缩并
        TD = Node.contract(T1, [tag1], T2, [tag2],
                           {i:"__1.%s"%i for i in T1.tags if i is not tag1},
                           {i:"__2.%s"%i for i in T2.tags if i is not tag2})
        tmp = TD.tags
        HH = Node(["__1", "__2", "__1.%s"%phy1, "__2.%s"%phy2], [l1, l2, l1, l2], H)
        TD = Node.contract(TD, ["__1.%s"%phy1, "__2.%s"%phy2], HH, ["__1", "__2"])
        TD.transpose(tmp)
        # SVD
        TD1, TD2 = Node.svd(TD, len(T1.tags)-1, tag1, tag2, cut)
        TD1.rename_leg({"__1.%s"%i:i for i in T1.tags if i is not tag1})
        TD2.rename_leg({"__2.%s"%i:i for i in T2.tags if i is not tag2})
        TD1.transpose(T1.tags)
        TD2.transpose(T2.tags)
        T1.replace(TD1)
        T2.replace(TD2)

    @staticmethod
    def qr_update(T1, T2, tag1, tag2, phy1, phy2, H, cut=None):
        """Update two tensor with Hamiltonian

        update in Simple Update method

        Args:
            T1, T2: tensor wait to be update
            phy1, phy2: the physical dimension of T1, T2
            tag1, tag2: the dimension wait to be contracted
            H: Hamiltonian
            cut: the number of singularvalue remain
        """
        # 准备
        l1 = T1.dims[T1.tags.index(phy1)]
        l2 = T2.dims[T2.tags.index(phy2)]
        if cut is None:
            cut = T1.dims[T1.tags.index(tag1)]
        # 缩并
        TD = Node.contract(T1, [tag1], T2, [tag2],
                           {i:"__1.%s"%i for i in T1.tags if i is not tag1},
                           {i:"__2.%s"%i for i in T2.tags if i is not tag2})
        tmp = TD.tags
        HH = Node(["__1", "__2", "__1.%s"%phy1, "__2.%s"%phy2], [l1, l2, l1, l2], H)
        TD = Node.contract(TD, ["__1.%s"%phy1, "__2.%s"%phy2], HH, ["__1", "__2"])
        TD.transpose(tmp)
        # SVD
        TD1, TD2 = Node.svd(TD, len(T1.tags)-1, tag1, tag2, cut)
        TD1.rename_leg({"__1.%s"%i:i for i in T1.tags if i is not tag1})
        TD2.rename_leg({"__2.%s"%i:i for i in T2.tags if i is not tag2})
        TD1.transpose(T1.tags)
        TD2.transpose(T2.tags)
        T1.replace(TD1)
        T2.replace(TD2)
