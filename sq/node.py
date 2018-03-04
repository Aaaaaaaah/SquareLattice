import numpy as np

class Node(object):

    @staticmethod
    def check_dims(dims):
        try:
            tmp_dims = list(dims)
        except Exception:
            raise Exception("dims cannot be converted to list")
        for i in tmp_dims:
            assert isinstance(i, int), 'dim should be int'
        return tmp_dims

    @staticmethod
    def check_tags(tags, length):
        try:
            tmp_tags = list(tags)
        except Exception:
            raise Exception("tags cannot be converted to list")
        for i in tmp_tags:
            assert isinstance(i, str), 'tag should be str'
        assert len(set(tmp_tags)) == len(tmp_tags), 'tags should not duplicated'
        assert len(tmp_tags) == length, 'invalid length of tags'
        return tmp_tags

    def __init__(self, tags, dims, data=None, init_data=True):
        self.dims = self.check_dims(dims)
        self.tags = self.check_dims(tags, len(self.dims))

        if init_data:
            if data:
                try:
                    self.data = np.reshape(np.array(data, np.float32), self.dims)
                except Exception as e:
                    raise Exception("cannot init node data") from e
            else:
                self.data = np.random.random(self.dims)
        else:
            self.data = None

    @classmethod
    def copy_shape(cls, tensor):
        return cls(tensor.tags, tensor.dims, data=None, init_data=False)

    def reshape(self, tags, dims=None):
        self.dims = self.check_dims(dims)
        self.tags = self.check_dims(tags, len(self.dims))
        if self.data:
            self.data = np.reshape(self.data, self.dims)

    def __repr__(self):
        return "Node with dims: %s"%str(zip(self.tags, self.dims))

    def rename_leg(self, new_tags):
        if isinstance(new_tags, list):
            self.tags = self.check_tags(new_tags, len(self.tags))

        elif isinstance(new_tags, dict):
            for i, j in new_tags.items():
                try:
                    self.tags[self.tags.index(i)] = j
                except Exception as e:
                    raise Exception("error when replace tag") from e

        else:
            raise Exception("new_tags must be list or dict")

    #转置
    @classmethod
    def transpose(cls, tensor, tags):
        tags = cls.check_tags(tags, len(tensor.tags))
        data = np.transpose(tensor.data, [tensor.tags.index(i) for i in tags])
        dims = [tensor.dims[tensor.tags.index(i)] for i in tags]
        result = cls(tags, dims, None, False)
        result.data = data
        return result

    #张量操作
    @classmethod
    def contract(cls, tensor1, tags1, tensor2, tags2, tags_dict1=None, tags_dict2=None):
        if tags_dict1 is None:
            tags_dict1 = {}
        else:
            assert isinstance(tags_dict1, dict)
        if tags_dict2 is None:
            tags_dict2 = {}
        else:
            assert isinstance(tags_dict2, dict)

        # order:the indexs of legs waiting for contracting
        try:
            order1 = [tensor1.tags.index(i) for i in tags1]
            order2 = [tensor2.tags.index(i) for i in tags2]
        except Exception as e:
            raise Exception("tag to contract not match") from e

        for i in tags_dict1:
            assert i in tensor1.tags and i not in tags1, "tag to change not match"
        for i in tags_dict2:
            assert i in tensor2.tags and i not in tags2, "tag to change not match"

        # generate the contribute of the answer
        tags = [j if j not in tags_dict1 else tags_dict1[j] \
                for i, j in enumerate(tensor1.tags) if i not in order1] +\
               [j if j not in tags_dict2 else tags_dict2[j] \
                for i, j in enumerate(tensor2.tags) if i not in order2]
        dims = [j for i, j in enumerate(tensor1.dims) if i not in order1] +\
                [j for i, j in enumerate(tensor2.dims) if i not in order2]

        #initiate the answer
        data = np.tensordot(tensor1.data, tensor2.data, [order1, order2])
        result = cls(tags, dims, None, False)
        result.data = data
        return result

    @classmethod
    def svd(cls, tensor, tags, tag1, tag2, cut=None):
        assert isinstance(tag1, str), "tag should be str"
        assert isinstance(tag2, str), "tag should be str"
        assert tag1 not in tensor.tags, "duplicated tags"
        assert tag2 not in tensor.tags, "duplicated tags"

        if isinstance(tags, list):
            for i in tags:
                assert i in tensor.tags, "invalid tags"
            assert len(tags) == len(set(tags)), "duplicated tags"

            num = len(tags)
            tags = tags + list(set(tensor.tags) - set(tags))
            tensor_transposed = cls.transpose(tensor, tags)
        elif isinstance(tags, int):
            num = tags
            tensor_transposed = tensor
        else:
            raise Exception("unrecognized tags format")

        dims1 = tensor_transposed.dims[:num]
        dims2 = tensor_transposed.dims[num:]
        data1, env, data2 = np.linalg.svd(
            np.reshape(
                tensor_transposed.data,
                [np.prod(dims1), np.prod(dims2)])
        )

        if not cut:
            cut = len(env)
        else:
            assert isinstance(cut, int), "cut dim should be int"
            assert cut <= len(env), "cut dim should be less than total dim"

        env = env[:cut]

        data1 = data1[:, :cut]
        tags1 = tensor_transposed.tags[:num] + [tag1]
        dims1 = dims1 + [cut]
        data1 = np.reshape(data1, dims1)

        data2 = data2[:cut, :]
        tags2 = [tag2] + tensor_transposed.tags[num:]
        dims2 = [cut] + dims2
        data2 = np.reshape(data2, dims2)

        tensor1 = cls(tags1, dims1, None, False)
        tensor1.data = np.reshape(data1, dims1)
        tensor2 = cls(tags2, dims2, None, False)
        tensor2.data = np.reshape(data2, dims2)
        return tensor1, env, tensor2

    @classmethod
    def qr(cls, tensor, tags, tag1, tag2, cut=None):
        assert isinstance(tag1, str), "tag should be str"
        assert isinstance(tag2, str), "tag should be str"
        assert tag1 not in tensor.tags, "duplicated tags"
        assert tag2 not in tensor.tags, "duplicated tags"

        if isinstance(tags, list):
            for i in tags:
                assert i in tensor.tags, "invalid tags"
            assert len(tags) == len(set(tags)), "duplicated tags"

            num = len(tags)
            tags = tags + list(set(tensor.tags) - set(tags))
            tensor_transposed = cls.transpose(tensor, tags)
        elif isinstance(tags, int):
            num = tags
            tensor_transposed = tensor
        else:
            raise Exception("unrecognized tags format")

        dims1 = tensor_transposed.dims[:num]
        dims2 = tensor_transposed.dims[num:]
        data1, data2 = np.linalg.qr(
            np.reshape(
                tensor_transposed.data,
                [np.prod(dims1), np.prod(dims2)])
        )

        if not cut:
            cut = data1.shape[1]
        else:
            assert isinstance(cut, int), "cut dim should be int"
            assert cut <= data1.shape[1], "cut dim should be less than total dim"

        data1 = data1[:, :cut]
        tags1 = tensor_transposed.tags[:num] + [tag1]
        dims1 = dims1 + [cut]
        data1 = np.reshape(data1, dims1)

        data2 = data2[:cut, :]
        tags2 = [tag2] + tensor_transposed.tags[num:]
        dims2 = [cut] + dims2
        data2 = np.reshape(data2, dims2)

        tensor1 = cls(tags1, dims1, None, False)
        tensor1.data = np.reshape(data1, dims1)
        tensor2 = cls(tags2, dims2, None, False)
        tensor2.data = np.reshape(data2, dims2)
        return tensor1, tensor2
