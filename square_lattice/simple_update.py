import numpy as np
from .node import Node

class SimpleNode(Node):
    def __init__(self, tags, dims, data=None, init_data=True, envs=None, init_envs=True):
        super().__init__(tags, dims, data, init_data)

        if envs:
            assert len(envs) == len(self.dims), "number of envs and dims should be the same"
            self.envs = []
            for i, j in zip(envs, self.dims):
                if i:
                    tmp = np.array(i, np.float32)
                    assert tmp.shape == (j,), "dims of envs not match dims"
                    self.envs.append(tmp)
                else:
                    self.envs.append(np.ones(j))
        else:
            if init_envs:
                self.envs = [np.ones(i) for i in self.dims]
            else:
                self.envs = [None for _ in self.dims]

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
    def absorb_envs(tensor, power, tags=None, exclude=False):
        ans = tensor.data.copy()
        if not tags:
            tags = range(len(tensor.dims))
            if exclude:
                raise Exception("exclude must be false when tags is none")
        else:
            tags = [tensor.tags[i] if isinstance(i, int) else i for i in tags]
            if(exclude):
                tags = list(set(tensor.tags) - set(tags))
            tags = [tensor.tags.index(i) for i in tags]
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
        tmp_tensor.envs = tensor.envs
        ans = super().svd(tmp_tensor, tags, tag1, tag2, cut)
        ans[0].envs = ans[3].envs[:ans[4]] + [np.ones(ans[0].dims[-1])]
        ans[2].envs = [np.ones(ans[2].dims[0])] + ans[3].envs[ans[4]:]
        ans[0].data = cls.absorb_envs(ans[0], -2)
        ans[2].data = cls.absorb_envs(ans[2], -2)
        ans[0].envs[-1] = ans[1]
        ans[2].envs[0] = ans[1]
        return ans[0], ans[2], ans[3], ans[4]

    @classmethod
    def update(cls, tensor1, tensor2, tag1, tag2, phy1, phy2, hamiltonian, cut=None):
        len1 = tensor1.dims[tensor1.tags.index(phy1) if isinstance(phy1, str) else int(phy1)]
        len2 = tensor2.dims[tensor2.tags.index(phy2) if isinstance(phy2, str) else int(phy2)]
        tmp_tensor = cls.contract(
            tensor1, [tag1], tensor2, [tag2],
            {i:"__1.%s"%i for i in tensor1.tags if i is not tag1},
            {i:"__2.%s"%i for i in tensor2.tags if i is not tag2}
        )
        tmp_tags = tmp_tensor.tags
        hamiltonian_tensor = cls(
            ["__1", "__2", "__1.%s"%phy1, "__2.%s"%phy2],
            [len1, len2, len1, len2],
            hamiltonian
        )
        total_tensor = cls.contract(
            tmp_tensor,
            ["__1.%s"%phy1, "__2.%s"%phy2],
            hamiltonian_tensor,
            ["__1", "__2"]
        )
        ans_tensor = cls.transpose(total_tensor, tmp_tags)
        new_tensor1, new_tensor2, _, _ = cls.svd(ans_tensor, len(tensor1.tags)-1, tag1, tag2, cut)
        new_tensor1.rename_leg({"__1.%s"%i:i for i in tensor1.tags if i is not tag1})
        new_tensor2.rename_leg({"__2.%s"%i:i for i in tensor2.tags if i is not tag2})

        ans_tensor1 = cls.transpose(new_tensor1, tensor1.tags)
        ans_tensor2 = cls.transpose(new_tensor2, tensor2.tags)
        tensor1.data = ans_tensor1.data
        tensor1.envs = ans_tensor1.envs
        tensor1.dims = ans_tensor1.dims
        tensor2.data = ans_tensor2.data
        tensor2.envs = ans_tensor2.envs
        tensor2.dims = ans_tensor2.dims

    @classmethod
    def qr_update(cls, tensor1, tensor2, tag1, tag2, phy1, phy2, hamiltonian, cut=None):

        len1 = tensor1.dims[tensor1.tags.index(phy1) if isinstance(phy1, str) else int(phy1)]
        len2 = tensor2.dims[tensor2.tags.index(phy2) if isinstance(phy2, str) else int(phy2)]
        hamiltonian_tensor = cls(
            ["__1", "__2", "__1.%s"%phy1, "__2.%s"%phy2],
            [len1, len2, len1, len2],
            hamiltonian
        )
        tmp_tensor1 = cls.copy_shape(tensor1)
        tmp_tensor2 = cls.copy_shape(tensor2)
        tmp_tensor1.data = cls.absorb_envs(tensor1, 2, [tag1, phy1], exclude=True)
        tmp_tensor2.data = cls.absorb_envs(tensor2, 2, [tag2, phy2], exclude=True)

        q1, r1 = cls.qr(tmp_tensor1, list(set(tmp_tensor1.tags)-set([tag1, phy1])), tag1, "__1.%s"%tag1)
        q2, r2 = cls.qr(tmp_tensor2, list(set(tmp_tensor2.tags)-set([tag2, phy2])), tag2, "__2.%s"%tag2)

        r1.envs[r1.tags.index(tag1)] = tensor1.envs[tensor1.tags.index(tag1)]
        r2.envs[r2.tags.index(tag2)] = tensor2.envs[tensor2.tags.index(tag2)]
        prod_tensor = cls.contract(r1, [tag1], r2, [tag2], {phy1: "__1.%s"%phy1}, {phy2: "__2.%s"%phy2})

        total_tensor = cls.contract(
            prod_tensor,
            ["__1.%s"%phy1, "__2.%s"%phy2],
            hamiltonian_tensor,
            ["__1", "__2"]
        )
        svd_tensor1, svd_tensor2, _, _ = cls.svd(total_tensor, ["__1.%s"%tag1, "__1.%s"%phy1], tag1, tag2, cut=cut)
        new_tensor1 = cls.contract(svd_tensor1, ["__1.%s"%tag1], q1, [tag1], {"__1.%s"%phy1:phy1}, {})
        new_tensor2 = cls.contract(svd_tensor2, ["__2.%s"%tag2], q2, [tag2], {"__2.%s"%phy2:phy2}, {})

        ans_tensor1 = cls.transpose(new_tensor1, tensor1.tags)
        ans_tensor2 = cls.transpose(new_tensor2, tensor2.tags)
        tensor1.data = ans_tensor1.data
        tensor1.data = cls.absorb_envs(tensor1, -2, [tag1], exclude=True)
        tensor1.envs[tensor1.tags.index(tag1)] = ans_tensor1.envs[ans_tensor1.tags.index(tag1)]
        tensor1.dims = ans_tensor1.dims
        tensor2.data = ans_tensor2.data
        tensor2.data = cls.absorb_envs(tensor2, -2, [tag2], exclude=True)
        tensor2.envs[tensor2.tags.index(tag2)] = ans_tensor2.envs[ans_tensor2.tags.index(tag2)]
        tensor2.dims = ans_tensor2.dims
