# -*- coding: utf-8 -*-

"""
    module config
"""
import copy
import numpy as np

class config(object):
    def __init__(self, n, m, data=None):
        self.rows = n
        self.cols = m
        if data==None:
            self.spins = [[0 for _ in range(m)] for _ in range(n)]
        else:
            self.spins = copy.deepcopy(data)

    def attempt_step(self):
        new_n = np.random.randint(self.rows)
        new_m = np.random.randint(self.cols)
        new_spins = copy.deepcopy(self.spin)
        new_spins[new_n][new_m] = 1 - new_spin[new_n][new_m]
        return new_spin

    def evolve(self):
        pass
