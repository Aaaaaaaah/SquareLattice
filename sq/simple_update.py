import numpy as np
from .node import Node

class SimpleNode(Node):
    def __init__(self, tags, dims, data=None, init_data=True, envs=None, init_envs=False):
        super().__init__(tags, dims, data, init_data)

        if init_envs:
            if envs:
                assert len(envs) == len(self.dims), "number of envs and dims should be the same"
                self.envs = []
                for i, j in zip(envs, self.dims):
                    if i:
                        tmp = np.array(i, np.float32)
                        assert tmp.shape == (j,), "dims of envs not match dims"
                        self.envs.push(tmp)
                    else:
                        self.envs.push(np.ones(j))
            else:
                envs = [np.ones(i) for i in range(self.dims)]
        else:
            self.envs = [None for _ in range(self.dims)]

    @classmethod
    def absorb(cls, tensor, tags=None):
        ans = super().copy_shape(tensor)
        ans.data = cls.absorb_envs(tensor, 1, tags)
        return ans

    @staticmethod
    def connect(tensor1, tag1, tensor2, tag2):
        if isinstance(tag1, str):
            tag1 = tensor1.tags.index(tag1)
        if isinstance(tag2, str):
            tag2 = tensor2.tags.index(tag2)
        assert isinstance(tag1, int)
        assert isinstance(tag2, int)
        tensor1.envs[tag1] = tensor2.envs[tag2]

    @staticmethod
    def absorb_envs(tensor, power, tags=None):
        ans = tensor.data.copy()
        if not tags:
            tags = range(len(tensor.dims))
        else:
            for i, j in enumerate(tags):
                if isinstance(j, str):
                    tags[i] = tensor.tags.index(j)
        for i in tags:
            tmp = np.ones(len(tensor.dims), dtype=int)
            tmp[i] = tensor.dims[i]
            ans *= np.reshape(np.power(tensor.envs[i], power), tmp)
        return ans

    @classmethod
    def transpose(cls, tensor, tags):
        envs = [tensor.envs[tensor.tags.index(i)] for i in tags]
        result = super().transpose(tensor, tags)
        result.envs = envs
        return result

    @classmethod
    def contract(cls, tensor1, tags1, tensor2, tags2, tags_dict1=None, tags_dict2=None):
        tmp_tensor1 = cls.copy_shape(tensor1)
        tmp_tensor2 = cls.copy_shape(tensor2)
        tmp_tensor1.data = cls.absorb_envs(tensor1, 1, tags1)
        tmp_tensor2.data = cls.absorb_envs(tensor2, 1, tags2)
        ans = super().contract(tmp_tensor1, tags1, tmp_tensor2, tags2, tags_dict1, tags_dict2)
        ans.envs = [j for i, j in zip(tensor1.tags, tensor1.envs) if i not in tags1] +\
                    [j for i, j in zip(tensor2.tags, tensor2.envs) if i not in tags2]
        return ans

    @classmethod
    def svd(cls, tensor, tags, tag1, tag2, cut=None):
        tmp_tensor = cls.copy_shape(tensor)
        tmp_tensor.data = cls.absorb_envs(tensor, 2)
        ans = super.svd(tmp_tensor, tags, tag1, tag2, cut)
        ans[0].envs = ans[3].envs[:ans[4]] + [np.ones(ans[0].dims[-1])]
        ans[2].envs = [np.ones(ans[2].dims[0])] + ans[3].envs[ans[4]:]
        ans[0].data = cls.absorb_envs(ans[0], -2)
        ans[2].data = cls.absorb_envs(ans[2], -2)
        ans[0].envs[-1] = ans[1]
        ans[2].envs[0] = ans[1]
        return ans[0], ans[2]

    @classmethod
    def update(cls):
        pass
