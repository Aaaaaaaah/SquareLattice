# -*- coding: utf-8 -*- #
"""
This module include lattice node
"""

import numpy as np
from node import Node

class lattice(object):
    """Lattice in a tensor network

    A square like lattice only for now, every Node in this lattice
    have "w","a","s","d" which indicate "up","left","down","right".
    For simplicity, we assume only and all these four directions have
    connections.

    Attribute:
        rows, cols: how many rows or columns
        data: matrix of Node Class
        bound_cond: represent the boundary condition, but the form undecided yet
            Temporarily use boolean, True is limited boundary
    """

    def __init__(self, rows, cols, data=None, bound_cond=True):
        pass

    def update(times):
        pass

    def energy()
