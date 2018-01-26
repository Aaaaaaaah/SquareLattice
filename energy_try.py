# -*- coding: utf-8 -*-
import time
import numpy as np
from node import Node
from energy import square_lattice
import sl

def time_count(func_name, func, *args):
    print(func_name, ":")
    time_start = time.time()
    func(*args)
    time_end = time.time()
    print("time = ", time_end - time_start)
    print()

A = square_lattice(sl.lattice, "p", 4, 4, sl.H)

time_count("WS", A.calc_cnfig_energy)

for i, j in enumerate(A.memory):
    print(i, j)
