# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from node import Node

L1 = 4
L2 = 4
D = 4

#   u
# l   r
#   d

def _node(s):
    l = [D]*len(s)
    l.append(2)
    return Node(list(s)+["p"],l)

def node_in_lattice(i,j):
    if i == 0:
        if j == 0:
            return _node("dr")
        elif j != L2-1:
            return _node("ldr")
        else:
            return _node("ld")
    elif i != L1:
        if j == 0:
            return _node("urd")
        elif j != L2-1:
            return _node("ulrd")
        else:
            return _node("uld")
    else:
        if j == 0:
            return _node("ur")
        elif j !=  L2-1:
            return _node("ulr")
        else:
            return _node("ul")

lattice = [[node_in_lattice(i,j) for j in range(L2)] for i in range(L1)]

ep = 0.1

H = np.reshape([[0.25,0,0,0],[0,-0.25,0.5,0],[0,0.5,-0.25,0],[0,0,0,0.25]],[2,2,2,2])
I = np.reshape(np.identity(4,dtype=np.float32),[2,2,2,2])
expH = I - 4.*ep*H

for t in range(1):
    print t
    for i in range(0,L1):
        for j in range(0,L2-1,2):
            Node.update(lattice[i][j],lattice[i][j+1],"r","l","p","p",expH)
    for i in range(0,L1):
        for j in range(1,L2-1,2):
            Node.update(lattice[i][j],lattice[i][j+1],"r","l","p","p",expH)
    for j in range(0,L2):
        for i in range(0,L1-1,2):
            Node.update(lattice[i][j],lattice[i+1][j],"d","u","p","p",expH)
    for j in range(0,L2):
        for i in range(1,L1-1,2):
            Node.update(lattice[i][j],lattice[i+1][j],"d","u","p","p",expH)

config = tf.ConfigProto()
#config.device_count["GPU"] = 0
#config.gpu_options.allow_growth = True
#config.graph_options.optimizer_options.global_jit_level=tf.OptimizerOptions.ON_1
config.log_device_placement = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

sess = tf.Session()
print sess.run(lattice[1][1].envs[0],options=options, run_metadata=run_metadata)

fetched_timeline = timeline.Timeline(run_metadata.step_stats)
chrome_trace = fetched_timeline.generate_chrome_trace_format()
with open('timeline.json', 'w') as f:
    f.write(chrome_trace)
