# -*- coding: utf-8 -*-
"""
This module include class Node
"""

import numpy as np
import tensorflow as tf

import node

Node = node.Node

class J1Node(Node):
    """Node of the lattice

    Represent one node in the whole network and support necessary
    operations on node.

    Attribute:
        tags: the name of each dimension
        dims: the order of each dimension
        data: the tensor data of the node.
        envs: the environments of each dimension
    """

    # 初始化函数
    def __init__(self, tags, dims, data=None, envs=None, init_data=True):
        super(J1Node, self).__init__(tags, dims, data, init_data)
        if envs is None:
            envs = [None for _ in tags]
        self.envs = []
        if init_data:
            if node.BASE == "NP":
                for i, j in zip(dims, envs):
                    if j is None:
                        tmp = np.ones(i)
                    else:
                        tmp = np.array(j, dtype=np.float32)
                        assert tmp.shape == (i,)
                    self.envs.append(tmp)
            elif node.BASE == "TF":
                for i, j in zip(dims, envs):
                    if j is None:
                        tmp = tf.ones(i)
                    else:
                        tmp = tf.cast(j, dtype=tf.float32)
                        assert tmp.shape == (i,)
                    self.envs.append(tmp)
            else:
                raise Exception("FT is TODO")

    def replace(self,other):
        self.data = other.data
        self.envs = other.envs
        self.dims = other.dims
        self.tags = other.tags

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
        if node.BASE == "NP":
            ans = tensor.data.copy()
        elif node.BASE == "TF":
            ans = tensor.data
        else:
            raise Exception("FT is TODO")

        if legs is None:
            legs = range(len(tensor.dims))

        for i in legs:
            tmp = np.ones(len(tensor.dims), dtype=int)
            tmp[i] = tensor.dims[i]
            if node.BASE == "NP":
                ans *= np.reshape(np.power(tensor.envs[i], pows), tmp)
            elif node.BASE == "TF":
                temp = tf.reshape(tensor.envs[i], tmp)
                if pows > 0:
                    for _ in range(pows):
                        ans *= temp
                else:
                    for _ in range(-pows):
                        ans /= temp
        return ans

    def normize(self):
        if node.BASE == "NP":
            self.data /= np.max(np.abs(self.data))
        elif node.BASE == "TF":
            self.data /= tf.reduce_max(tf.abs(self.data))
        else:
            raise Exception("FT is TODO")

    #张量操作

    # simple update
    @staticmethod
    def update(tensor1, tensor2, tag1, tag2, phy1, phy2, hamiltonian, cut=None):
        """Update two tensor with Hamiltonian

        update in Simple Update method

        Args:
            tensor1, tensor2: tensor wait to be update
            phy1, phy2: the physical dimension of tensor1, tensor2
            tag1, tag2: the dimension wait to be contracted
            H: Hamiltonian
            cut: the number of singularvalue remain
        """
        # 准备
        l1 = tensor1.dims[tensor1.tags.index(phy1)]
        l2 = tensor2.dims[tensor2.tags.index(phy2)]
        if cut is None:
            cut = tensor1.dims[tensor1.tags.index(tag1)]
        # 缩并
        tensor_res = Node.contract(
            tensor1,
            [tag1],
            tensor2,
            [tag2],
            {i:"__1.%s"%i for i in tensor1.tags if i is not tag1},
            {i:"__2.%s"%i for i in tensor2.tags if i is not tag2}
        )
        tmp = tensor_res.tags
        hamiltonian_tensor = Node(["__1", "__2", "__1.%s"%phy1, "__2.%s"%phy2], [l1, l2, l1, l2], hamiltonian)
        tensor_res = Node.contract(tensor_res, ["__1.%s"%phy1, "__2.%s"%phy2], hamiltonian_tensor, ["__1", "__2"])
        tensor_res = Node.transpose(tensor_res, tmp)
        # SVD
        tensor_res1, env, tensor_res2 = Node.svd(tensor_res, len(tensor1.tags)-1, tag1, tag2, cut)
        ### 这个各种环境没有处理
        tensor_res1.rename_leg({"__1.%s"%i:i for i in tensor1.tags if i is not tag1})
        tensor_res2.rename_leg({"__2.%s"%i:i for i in tensor2.tags if i is not tag2})
        tensor_res1 = Node.transpose(tensor_res1, tensor1.tags)
        tensor_res2 = Node.transpose(tensor_res2, tensor2.tags)
        tensor1.replace(tensor_res1)
        tensor2.replace(tensor_res2)

    @staticmethod
    def qr_update(tensor1, tensor2, tag1, tag2, phy1, phy2, hamitonian, cut=None):
        """Update two tensor with Hamiltonian

        update in Simple Update method

        Args:
            tensor1, tensor2: tensor wait to be update
            phy1, phy2: the physical dimension of tensor1, tensor2
            tag1, tag2: the dimension wait to be contracted
            H: Hamiltonian
            cut: the number of singularvalue remain
        """
        # 准备
        l1 = tensor1.dims[tensor1.tags.index(phy1)]
        l2 = tensor2.dims[tensor2.tags.index(phy2)]
        if cut is None:
            cut = tensor1.dims[tensor1.tags.index(tag1)]
        # 缩并
        tensor_res = Node.contract(tensor1, [tag1], tensor2, [tag2],
                           {i:"__1.%s"%i for i in tensor1.tags if i is not tag1},
                           {i:"__2.%s"%i for i in tensor2.tags if i is not tag2})
        tmp = tensor_res.tags
        hamitonian_tensor = Node(["__1", "__2", "__1.%s"%phy1, "__2.%s"%phy2], [l1, l2, l1, l2], hamitonian)
        tensor_res = Node.contract(tensor_res, ["__1.%s"%phy1, "__2.%s"%phy2], hamitonian_tensor, ["__1", "__2"])
        tensor_res = Node.transpose(tensor_res, tmp)
        # SVD
        tensor_res1, tensor_res2 = Node.svd(tensor_res, len(tensor1.tags)-1, tag1, tag2, cut)
        tensor_res1.rename_leg({"__1.%s"%i:i for i in tensor1.tags if i is not tag1})
        tensor_res2.rename_leg({"__2.%s"%i:i for i in tensor2.tags if i is not tag2})
        tensor_res1 = Node.transpose(tensor_res1, tensor1.tags)
        tensor_res2 = Node.transpose(tensor_res2, tensor2.tags)
        tensor1.replace(tensor_res1)
        tensor2.replace(tensor_res2)
