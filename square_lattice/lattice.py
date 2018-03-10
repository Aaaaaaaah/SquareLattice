import numpy as np
from .node import Node
from .simple_update import SimpleNode

class Lattice(object):

    def __init__(self):
        self.node_pool = {}
        self.link_pool = []

    def add_node(self, name, tags, dims, data=None, init_data=True, envs=None, init_envs=True):
        if name is None:
            name = str(len(self.node_pool))
        tmp_node = SimpleNode(tags, dims, data, init_data, envs, init_envs)
        self.node_pool[name] = tmp_node
        return tmp_node

    def add_link(self, name1, tag1, name2, tag2):
        SimpleNode.connect(
            self.node_pool[name1], tag1,
            self.node_pool[name2], tag2
        )
        self.link_pool.append({"T1":name1, "t1":tag1, "T2":name2, "t2":tag2})

    def update(self, hamiltonian, cut=None, qr=False):
        if qr:
            update_func = SimpleNode.qr_update
        else:
            update_func = SimpleNode.update
        tmp1 = self.link_pool
        while tmp1:
            tmp2 = []
            used_node = set()
            for i in tmp1:
                if i["T1"] not in used_node and i["T2"] not in used_node:
                    update_func(
                        self.node_pool[i["T1"]],
                        self.node_pool[i["T2"]],
                        i["t1"], i["t2"], "p", "p",
                        hamiltonian, cut=cut
                    )
                    used_node.add(i["T1"])
                    used_node.add(i["T2"])
                else:
                    tmp2.append(i)
            tmp1 = tmp2
        for i, j in self.node_pool.items():
            j.normize()

    def energy(self):
        pass
