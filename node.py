# -*- coding: utf-8 -*-
"""
This module include class Node
"""

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
        envf: if it is using environments, envf is True, else False
        normf: if it is using autonormalization, normf is True, else False
    """

    # 初始化函数
    def __init__(self, tags, dims, data=None, envs=None, envf=True, normf=True):
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
            if normf:
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
                    if normf:
                        tmp /= np.max(np.abs(tmp))
                    assert tmp.shape == (i,)
                    self.envs.append(tmp)
        else:
            self.envs = [np.ones(i) for i in dims]
        self.dims = list(dims)
        self.tags = list(tags)
        self.__envf = envf
        self.__normf = normf

    def __repr__(self):
        return "Node with dims: %s"%str(zip(self.tags, self.dims))

    #复制与替代
    @staticmethod
    def copy(tensor):
        return Node(tensor.tags,
                    tensor.dims,
                    tensor.data,
                    tensor.envs,
                    tensor.__envf,
                    tensor.__normf)

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
        self.__envf = other.__envf
        self.__normf = other.__normf

    #重命名脚,吸收环境等基本操作
    def rename_leg(self, tag_dict):
        """Rename the dimensions

        Give the dimensions some other names.

        Args:
            tag_dict: the dictionary of old tags and new tags
                with format {old tags : new tags}
        """
        tmp = [self.tags.index(i) for i in tag_dict]
        for i in tmp:
            self.tags[i] = tag_dict[self.tags[i]]
        return self

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
            self.__envf = False

    @normf.setter
    def normf(self, value):
        if value:
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

    #转置
    def transpose(self, tags):
        """Transpose the tensor data of the Node

        Args:
            tags: new arrangement of the old dimension names
        """
        self.data = np.transpose(self.data, [self.tags.index(i) for i in tags])
        tmp = self.dims
        self.dims = [tmp[self.tags.index(i)] for i in tags]
        tmp = self.envs
        self.envs = [tmp[self.tags.index(i)] for i in tags]
        self.tags = tags

    #张量操作
    @staticmethod
    def contract(T1, tags1, T2, tags2, tags_dict1=None, tags_dict2=None):
        """Contract two Node together

        Contract two Node together into a big Node

        Args:
            T1: the first Node
            tags1: the dimensions wait to be contracted in T1
            T1: the second Node
            tags1: the dimensions wait to be contracted in T2
            tags_dict1, tags_dict2: if there are contradict names of dimension
                in T1 and T2, then use the dictionary to fix the contadiction.

        Returns:
            T: the result of the contraction(envf = True)
        """
        envf1 = T1.envf
        envf2 = T2.envf
        T1.envf = True
        T2.envf = True
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
        T = Node(tags, dims, np.tensordot(TD1, TD2, [order1, order2]), envs,
                 normf=T1.normf & T2.normf)
        T1.envf = envf1
        T2.envf = envf2
        return T

    @staticmethod
    def svd(tensor, num, tag1, tag2, cut=None):
        """SVD decomposition of Node

        SVD decomposition of Node and update the environments between these two
        Node. At the same time, this method only select the first 'cut' data.

        Args:
            tensor: the Node wait to be decomposed.
            num: the first 'num' dimensions to be decomposed into one tensor.
                the left into another tensor
            tag1: name of new dimension in the first tensor
            tag2: name of new dimension in the second tensor
            cut: the number of singularvalues remain

        Returns:
            T1, T2: the tensor generated by SVD
        """
        order = 2 if tensor.envf else 1
        dims1 = tensor.dims[:num]
        dims2 = tensor.dims[num:]
        data1, env, data2 = np.linalg.svd(
            np.reshape(
                Node.absorb_envs(tensor, order),
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
        T1 = Node(tags1, dims1, data1, envs1, tensor.envf)
        T2 = Node(tags2, dims2, data2, envs2, tensor.envf)
        T1.data = Node.absorb_envs(T1, -order, range(len(dims1)-1))
        T2.data = Node.absorb_envs(T2, -order, range(1, len(dims2)))
        return T1, T2

    @staticmethod
    def qr(tensor, num, tag1, tag2, cut=None):
        """QR decomposition

        Decompose a Node with QR decomposition and return q, r matrix.

        Args:
            tensor: the Node wait to be decomposed.
            num: the first 'num' dimensions to be decomposed into one tensor.
                the left into another tensor
            tag1: name of new dimension in the first tensor
            tag2: name of new dimension in the second tensor
            cut: the rank remain

        Returns:
            q, r: the Q and R matrix of QR decomposition in Node class format
        """
        order = 2 if tensor.envf else 1
        dims1 = tensor.dims[:num]
        dims2 = tensor.dims[num:]
        data1, data2 = np.linalg.qr(
            np.reshape(
                Node.absorb_envs(tensor, order),
                [np.prod(dims1), np.prod(dims2)]),
            mode='reduced'
        )
        ##if data2.shape[0]!=data2.shape[1]:
        ##    tmp1, tmp2 = data2.shape[0], data2.shape[1]
        ##    data1 = np.concatenate((data1, np.zeros([tmp1, tmp2-tmp1])), axis=1)
        ##    data2 = np.concatenate((data2, np.zeros([tmp2-tmp1, tmp2])), axis=0)
        if cut is None:
            cut = data1.shape[1]
        env = np.ones(cut)
        data1 = data1[:, :cut]
        data2 = data2[:cut, :]
        tags1 = tensor.tags[:num] + [tag1]
        tags2 = [tag2] + tensor.tags[num:]
        dims1 = dims1 + [cut]
        dims2 = [cut] + dims2
        envs1 = tensor.envs[:num] + [env]
        envs2 = [env] + tensor.envs[num:]
        T1, T2 = Node(tags1, dims1, data1, envs1, True, tensor.normf), \
                 Node(tags2, dims2, data2, envs2, True, tensor.normf)
        T1.data = Node.absorb_envs(T1, -order, range(len(dims1)-1))
        T2.data = Node.absorb_envs(T2, -order, range(1, len(dims2)))
        return T1, T2

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
