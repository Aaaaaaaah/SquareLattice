import numpy as np
from .node import Node

class SimpleNode(Node):
    def __init__(self, tags, dims, data=None, init_data=True, envs=None, init_envs=True):
        super().__init__(tags, dims, data, init_data)

        if init_envs:
            if envs:
                assert len(envs) == len(self.dims), "number of envs and dims should be the same"
                self.envs = []
                for i, j in zip(envs,self.dims):
                    if i:
                        tmp = np.array(i, np.float32)
                        assert tmp.shape = (j,), "dims of envs not match dims"
                        self.envs.push(tmp)
                    else:
                        self.envs.push(np.ones(j))
            else:
                envs = [np.ones(i) for i in range(self.dims)]
        else:
            self.envs = [None for _ in range(self.dims)]
