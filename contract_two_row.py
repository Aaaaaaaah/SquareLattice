import numpy as np

L = 10

#from right to leftt
#contract two rows into one row

##generate psi0(left, up, right)
psi0 = [np.random.random([d, D])] + [np.random.random([D, d, D]) for _ in range(L-2)] + [np.random.random([D,d])]

##generate Operator(left, up, down, right)
Operator = [np.random.random([d, d, D]) + [np.random.random([D, d, d, D]) for _ in range(L-2)] + [np.random.random([D, d, d])]

##psi_new = psi0
psi_new = [i for i in psi0]

##main part
for _ in range(100)
    #initiate
    ##left unitarinalize
    for i in range(L-1)
        if i==0:
            psi_new[i], r = np.linalg.qr(psi_new[i])
        else:
            psi_new[i], r = np.linalg.qr(psi_new[i].reshape([]))
        psi_new[i+1] = np.linalg.tensordot(r, psi_new[i+1], axes = ([1],[0]))
    for i in range(L)
        #generate psi_new[i]
        #update left & right
    pass

##compare ans
