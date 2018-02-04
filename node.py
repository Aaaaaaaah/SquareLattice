# -*- coding: utf-8 -*-
"""
This module include class Node
"""

import numpy as np
import tensorflow as tf

BASE = "NP" # "NP" "TF" "FT"

class Node(object):
    """Node of the lattice

    Represent one node in the whole network and support necessary
    operations on node.

    Attribute:
        tags: the name of each dimension
        dims: the order of each dimension
        data: the tensor data of the node.
    """

    # 初始化函数
    def __init__(self, tags, dims, data=None, init_data=True):
        """Initiate the Node

        Give the Node a tensor data and each dim environments.
        If data is not given by user, data would be randomly given.
        If environments are not given, environments would be set as 1.

        Args:
            tags: the name of each dimension
            dims: the order of each dimension
            data: the tensor data of the node.
        """
        assert len(tags) == len(dims) # tags 和 dims 长度需要一样
        assert len(set(tags)) == len(tags) # tags没有 重复
        self.dims = list(dims)
        self.tags = list(tags)

        if init_data:
            if BASE == "NP":
                if data:
                    self.data = np.reshape(np.array(data, np.float32), self.dims)
                else:
                    self.data = np.random.random(self.dims)
            elif BASE == "TF":
                if data:
                    self.data = tf.reshape(tf.cast(data, tf.float32), self.dims)
                else:
                    self.data = tf.random_uniform(self.dims)
            else:
                raise Exception("FT is TODO")

    def __repr__(self):
        return "Node with dims: %s"%str(zip(self.tags, self.dims))

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

    #运算

    #转置
    @staticmethod
    def transpose(tensor, tags):
        """Transpose the tensor data of the Node

        Args:
            tags: new arrangement of the old dimension names
        """
        if BASE == "NP":
            data = np.transpose(tensor.data, [tensor.tags.index(i) for i in tags])
        elif BASE == "TF":
            data = tf.transpose(tensor.data, [tensor.tags.index(i) for i in tags])
        else:
            raise Exception("FT is TODO")
        dims = [tensor.dims[tensor.tags.index(i)] for i in tags]
        result = Node(tags, dims, None, False)
        result.data = data
        return result

    #张量操作
    @staticmethod
    def contract(tensor1, tags1, tensor2, tags2, tags_dict1=None, tags_dict2=None):
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
        if tags_dict1 is None:
            tags_dict1 = {}
        if tags_dict2 is None:
            tags_dict2 = {}
        # order:the indexs of legs waiting for contracting
        order1 = [tensor1.tags.index(i) for i in tags1]
        order2 = [tensor2.tags.index(i) for i in tags2]
        # generate the contribute of the answer
        for i in tags_dict1:
            assert i in tensor1.tags and i not in tags1
        for i in tags_dict2:
            assert i in tensor2.tags and i not in tags2
        tags = [j if j not in tags_dict1 else tags_dict1[j] \
                for i, j in enumerate(tensor1.tags) if i not in order1] +\
               [j if j not in tags_dict2 else tags_dict2[j] \
                for i, j in enumerate(tensor2.tags) if i not in order2]
        dims = [j for i, j in enumerate(tensor1.dims) if i not in order1] +\
                [j for i, j in enumerate(tensor2.dims) if i not in order2]
        #initiate the answer
        if BASE == "NP":
            data = np.tensordot(tensor1.data, tensor2.data, [order1, order2])
        elif BASE == "TF":
            data = tf.tensordot(tensor1.data, tensor2.data, [order1, order2])
        else:
            raise Exception("FT is TODO")
        result = Node(tags, dims, None, False)
        result.data = data
        return result

    @staticmethod
    def svd(tensor, tags, tag1, tag2, cut=None):
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
        for i in tags:
            assert i in tensor.tags
        assert len(tags) == len(set(tags))
        num = len(tags)
        tags = tags + list(set(tensor.tags) - set(tags))
        tensor_transposed = Node.transpose(tensor, tags)
        dims1 = tensor_transposed.dims[:num]
        dims2 = tensor_transposed.dims[num:]
        if BASE == "NP":
            data1, env, data2 = np.linalg.svd(
                np.reshape(
                    tensor_transposed.data,
                    [np.prod(dims1), np.prod(dims2)])
            )
            if not cut:
                cut = len(env)
            env = env[:cut]
            data1 = data1[:, :cut]
            tags1 = tensor_transposed.tags[:num] + [tag1]
            dims1 = dims1 + [cut]
            data1 = np.reshape(data1, dims1)
            data2 = data2[:cut, :]
            tags2 = [tag2] + tensor_transposed.tags[num:]
            dims2 = [cut] + dims2
            data2 = np.reshape(data2, dims2)
        elif BASE == "TF":
            env, data1, data2 = tf.svd(
                np.reshape(
                    tensor_transposed.data,
                    [np.prod(dims1), np.prod(dims2)])
            )
            if not cut:
                cut = len(env)
            env = env[:cut]
            data1 = data1[:, :cut]
            tags1 = tensor_transposed.tags[:num] + [tag1]
            dims1 = dims1 + [cut]
            data1 = tf.reshape(data1, dims1)
            data2 = data2[:, :cut]
            tags2 = tensor_transposed.tags[num:] + [tag2]
            dims2 = dims2 + [cut]
            data2 = tf.reshape(data2, dims2)
        else:
            raise Exception("FT is TODO")
        tensor1 = Node(tags1, dims1, None, False)
        tensor1.data = data1
        tensor2 = Node(tags2, dims2, None, False)
        tensor2.data = data2
        return tensor1, env, tensor2

    @staticmethod
    def qr(tensor, tags, tag1, tag2, cut=None):
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
        for i in tags:
            assert i in tensor.tags
        assert len(tags) == len(set(tags))
        num = len(tags)
        tags = tags + list(set(tensor.tags) - set(tags))
        tensor_transposed = Node.transpose(tensor, tags)
        dims1 = tensor_transposed.dims[:num]
        dims2 = tensor_transposed.dims[num:]
        if BASE == "NP":
            data1, data2 = np.linalg.qr(
                np.reshape(
                    tensor_transposed.data,
                    [np.prod(dims1), np.prod(dims2)])
            )
            if not cut:
                cut = len(data2)
            data1 = data1[:, :cut]
            tags1 = tensor_transposed.tags[:num] + [tag1]
            dims1 = dims1 + [cut]
            data1 = np.reshape(data1, dims1)
            data2 = data2[:cut, :]
            tags2 = [tag2] + tensor_transposed.tags[num:]
            dims2 = [cut] + dims2
            data2 = np.reshape(data2, dims2)
        elif BASE == "TF":
            data1, data2 = tf.qr(
                np.reshape(
                    tensor_transposed.data,
                    [np.prod(dims1), np.prod(dims2)])
            )
            if not cut:
                cut = data1.shape[1]
            data1 = data1[:, :cut]
            tags1 = tensor_transposed.tags[:num] + [tag1]
            dims1 = dims1 + [cut]
            data1 = tf.reshape(data1, dims1)
            data2 = data2[:, :cut]
            tags2 = tensor_transposed.tags[num:] + [tag2]
            dims2 = dims2 + [cut]
            data2 = tf.reshape(data2, dims2)
        else:
            raise Exception("FT is TODO")
        tensor1 = Node(tags1, dims1, None, False)
        tensor1.data = data1
        tensor2 = Node(tags2, dims2, None, False)
        tensor2.data = data2
        return tensor1, tensor2
