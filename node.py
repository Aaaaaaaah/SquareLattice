# -*- coding: utf-8 -*-

import numpy as np

class Node(object):
    """Node of the lattice

    Represent one node in the whole network and support necessary
    operations on node.

    Attribute:
        tags: the name of each dimension
        dims: the order of each dimension
        data: the tensor data of the node.
        envs: the environments of each dimension
    """

    def __init__(self, tags, dims, data=None, envs=None):
        """Initiate the Node

        Give the Node a tensor data and each dim environments.
        If data is not given by user, data would be randomly given.
        If environments are not given, environments would be set as 1.

        Args:
            tags: the name of each dimension
            dims: the order of each dimension
            data: the tensor data of the node.
            envs: the environments of each dimension
        """
        assert len(tags) == len(dims)
        assert len(set(tags)) == len(tags)
        if data is not None:
            self.data = np.reshape(np.array(data, dtype=np.float64), dims)
            self.data /= np.max(np.abs(self.data))
        else:
            self.data = np.random.random(dims)
        if envs is not None:
            self.envs = []
            for i, j in zip(dims, envs):
                if j is None:
                    self.envs.append(np.ones(i))
                else:
                    tmp = np.array(j, dtype=np.float64)
                    tmp /= np.max(np.abs(tmp))
                    assert tmp.shape == (i,)
                    self.envs.append(tmp)
        else:
            self.envs = [np.ones(i) for i in dims]
        self.dims = list(dims)
        self.tags = list(tags)

    def replace(self, other):
        """Replace itself with another node.

        A copy method, copy all attributes from the other node
        into this node.

        Args:
            other: another node object
        """
        self.data = other.data
        self.envs = other.envs
        self.dims = other.dims
        self.tags = other.tags

    def rename_leg(self, tag_dict):
        for i, j in tag_dict.items():
            self.tags[self.tags.index(i)] = j

    @staticmethod
    def absorb_envs(tensor, pows, legs=None):
        ans = tensor.data.copy()
        if legs is None:
            legs = range(len(tensor.dims))
        for i in legs:
            tmp = np.ones(len(tensor.dims), dtype=int)
            tmp[i] = tensor.dims[i]
            ans *= np.reshape(np.power(tensor.envs[i], pows), tmp)
        return ans

    def transpose(self, tags):
        self.data = np.transpose(self.data, [self.tags.index(i) for i in tags])
        tmp = self.dims
        self.dims = [tmp[self.tags.index(i)] for i in tags]
        tmp = self.envs
        self.envs = [tmp[self.tags.index(i)] for i in tags]
        self.tags = tags

    def __repr__(self):
        return "Node with dims: %s"%str(zip(self.tags, self.dims))

    @staticmethod
    def contract(T1, tags1, T2, tags2, tags_dict1=None, tags_dict2=None):
        if tags_dict1 is None:
            tags_dict1 = {}
        if tags_dict2 is None:
            tags_dict2 = {}
        # order:the indexs of legs waiting for contracting
        order1 = [T1.tags.index(i) for i in tags1]
        order2 = [T2.tags.index(i) for i in tags2]
        # absorb environment
        TD1 = Node.absorb_envs(T1, 1, order1)
        TD2 = Node.absorb_envs(T2, 1, order2)
        # generate the contribute of the answer
        for i in tags_dict1:
            assert i in T1.tags and i not in tags1
        for i in tags_dict2:
            assert i in T2.tags and i not in tags2
        tags = [j if j not in tags_dict1 else tags_dict1[j] \
                for i, j in enumerate(T1.tags) if i not in order1] +\
               [j if j not in tags_dict2 else tags_dict2[j] \
                for i, j in enumerate(T2.tags) if i not in order2]
        dims = [j for i, j in enumerate(T1.dims) if i not in order1] +\
                [j for i, j in enumerate(T2.dims) if i not in order2]
        envs = [j for i, j in enumerate(T1.envs) if i not in order1] +\
                [j for i, j in enumerate(T2.envs) if i not in order2]
        #initiate the answer
        T = Node(tags, dims, np.tensordot(TD1, TD2, [order1, order2]), envs)
        return T

    @staticmethod
    def svd(tensor, num, tag1, tag2, cut):
        dims1 = tensor.dims[:num]
        dims2 = tensor.dims[num:]
        data1, env, data2 = np.linalg.svd(
            np.reshape(
                Node.absorb_envs(tensor, 2),
                [np.prod(dims1), np.prod(dims2)])
        )
        env = np.sqrt(env[:cut])
        data1 = data1[:, :cut]
        data2 = data2[:cut, :]
        tags1 = tensor.tags[:num] + [tag1]
        tags2 = [tag2] + tensor.tags[num:]
        dims1 = dims1 + [cut]
        dims2 = [cut] + dims2
        envs1 = tensor.envs[:num] + [env]
        envs2 = [env] + tensor.envs[num:]
        T1, T2 = Node(tags1, dims1, data1, envs1), Node(tags2, dims2, data2, envs2)
        T1.data = Node.absorb_envs(T1, -2, range(len(dims1)-1))
        T2.data = Node.absorb_envs(T2, -2, range(1, len(dims2)))
        return T1, T2

    @staticmethod
    def update(T1, T2, tag1, tag2, phy1, phy2, H, cut=None):
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
