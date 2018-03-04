# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class Node(object):

    # 初始化函数
    def __init__(self, tags, dims, data=None, init_data=True):
        assert len(tags) == len(dims) # tags 和 dims 长度需要一样
        assert len(set(tags)) == len(tags) # tags没有 重复
        self.dims = list(dims)
        self.tags = list(tags)

        if init_data:
            if data:
                self.data = np.reshape(np.array(data, np.float32), self.dims)
            else:
                self.data = np.random.random(self.dims)

    def __repr__(self):
        return "Node with dims: %s"%str(zip(self.tags, self.dims))

    #重命名脚,吸收环境等基本操作
    def rename_leg(self, tag_dict):
        tmp = [self.tags.index(i) for i in tag_dict]
        for i in tmp:
            self.tags[i] = tag_dict[self.tags[i]]
        return self

    #运算

    #转置
    @staticmethod
    def transpose(tensor, tags):
        data = np.transpose(tensor.data, [tensor.tags.index(i) for i in tags])
        dims = [tensor.dims[tensor.tags.index(i)] for i in tags]
        result = Node(tags, dims, None, False)
        result.data = data
        return result

    #张量操作
    @staticmethod
    def contract(tensor1, tags1, tensor2, tags2, tags_dict1=None, tags_dict2=None):
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
        data = np.tensordot(tensor1.data, tensor2.data, [order1, order2])
        result = Node(tags, dims, None, False)
        result.data = data
        return result

    @staticmethod
    def svd(tensor, tags, tag1, tag2, cut=None):
        if type(tags) != int:
            for i in tags:
                assert i in tensor.tags
            assert len(tags) == len(set(tags))
            num = len(tags)
            tags = tags + list(set(tensor.tags) - set(tags))
            tensor_transposed = Node.transpose(tensor, tags)
        else:
            num = tags
            tensor_transposed = tensor
        dims1 = tensor_transposed.dims[:num]
        dims2 = tensor_transposed.dims[num:]
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
        tensor1 = Node(tags1, dims1, None, False)
        tensor1.data = data1
        tensor2 = Node(tags2, dims2, None, False)
        tensor2.data = data2
        return tensor1, env, tensor2

    @staticmethod
    def qr(tensor, tags, tag1, tag2, cut=None):
        if type(tags) != int:
            for i in tags:
                assert i in tensor.tags
            assert len(tags) == len(set(tags))
            num = len(tags)
            tags = tags + list(set(tensor.tags) - set(tags))
            tensor_transposed = Node.transpose(tensor, tags)
        else:
            num = tags
            tensor_transposed = tensor
        dims1 = tensor_transposed.dims[:num]
        dims2 = tensor_transposed.dims[num:]
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
        tensor1 = Node(tags1, dims1, None, False)
        tensor1.data = data1
        tensor2 = Node(tags2, dims2, None, False)
        tensor2.data = data2
        return tensor1, tensor2
