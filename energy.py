# -*- coding: utf-8 -*-

"""
    try to support the energy calculation of square lattice
"""

import numpy as np
from node import Node
from tool import decompose_tool, very_simple_contract, unitarilize, \
                 attempt_step, to_bin

#hat
hat = [Node(["phy"], [2], data=np.array([1.0,0.0])), Node(["phy"], [2], data=np.array([0.0,1.0]))]

class square_lattice(object):
    """
        tensor_array
        lattice_type
        redu_tensor redutensor[i][j][0 or 1]
    """
    def __init__(self, arr, phy, rows, cols, H, spins=None):
        self.n = rows
        self.m = cols
        self.tensor_array = [[Node.copy(j) for j in i] for i in arr]
        for i in self.tensor_array:
            for j in i:
                j.envf = False
                j.normf = False
        self.redu_tensor = [[[Node.contract(j, [phy], hat[k], ["phy"]) \
                for k in range(2)] for j in i] for i in self.tensor_array]
        self.H = H.copy()
        if spins==None:
            self.spins = [[0 for _ in range(cols)] for _ in range(rows)]
        else:
            self.spins = spins
        self.memory = [None for _ in range(2 ** (rows * cols))]

    @staticmethod
    def contract_two_row(psi0, operator, left="l", up="u", down="d", right="r"):
        """
            psi0: left, up, right
            operator: left, up, down, right
        """
        delta = 5e-4
        delta_err = 1e-10
        L = len(psi0)
        ##disable normf
        for i in psi0 + operator:
            i.normf = False
        ##unitarilize
        psi_new = [Node.copy(i) for i in psi0]
        unitarilize(psi_new, left, right)
        ##initiate side
        tmp = Node.contract(psi0[L-1], [up], operator[L-1], [down], {left:"down"}, {left:"mid"})
        tmp = Node.contract(tmp, [up], psi_new[L-1], [up], {}, {left:"up"})
        side = [tmp]
        for i in range(L-2, 0, -1):
            tmp = Node.contract(tmp, ["down"], psi0[i], [right], {}, {left:"down"})
            tmp = Node.contract(tmp, ["mid", up], operator[i], [right, down], {}, {left:"mid"})
            tmp = Node.contract(tmp, ["up", up], psi_new[i], [right, up], {}, {left:"up"})
            side = [tmp] + side
        side = [None] + side
        ##target energy
        tmp = [[Node.copy(i).rename_leg({up:down}) for i in psi0], \
               [Node.copy(i).rename_leg({up:down, down:up}) for i in operator], operator, psi0]
        energy0 = very_simple_contract(tmp, 4, L, up, down, left, right)
        ##main part
        dir = 1
        dir_dict = {1:right, -1:left}
        pos = 0
        energy1 = very_simple_contract([[Node.copy(i).rename_leg({up:down}) for i in psi_new], \
                                        psi_new], 2, L, up, down, left, right)
        #print(energy0)
        while (abs(energy1-energy0)>abs(energy0)*delta):
            print(pos, energy1)
            in_range = (pos-dir) in range(L)
            if in_range:
                tmp = Node.contract(side[pos-dir], ["down"], psi0[pos], [dir_dict[-dir]], {}, {dir_dict[dir]:"down"})
                tmp = Node.contract(tmp, ["mid", up], operator[pos], [dir_dict[-dir], down], {}, {dir_dict[dir]:"mid"})
            else:
                tmp = Node.copy(psi0[pos]).rename_leg({dir_dict[dir]:"down"})
                tmp = Node.contract(tmp, [up], operator[pos], [down], {}, {dir_dict[dir]:"mid"})
            psi_new[pos] = Node.contract(tmp, ["mid", "down"], side[pos+dir], ["mid", "down"], {"up":dir_dict[-dir]} \
                                if in_range else {}, {"up":dir_dict[dir]})
            psi_new[pos],r = decompose_tool(Node.qr, psi_new[pos], dir_dict[dir], dir_dict[dir], dir_dict[-dir])
            psi_new[pos+dir] = Node.contract(r, [dir_dict[dir]], psi_new[pos+dir], [dir_dict[-dir]])
            side[pos] = Node.contract(tmp, ["up", up] if in_range else [up], psi_new[pos], \
                                      [dir_dict[-dir], up] if in_range else [up], {}, {dir_dict[dir]:"up"})
            pos += dir
            if pos in [0, L-1]:
                dir = -dir
            energy_tmp = very_simple_contract([[Node.copy(i).rename_leg({up:down}) for i in psi_new], \
                                            psi_new], 2, L, up, down, left, right)
            if abs(energy1-energy_tmp)<abs(energy1)*delta_err :
                break
            energy1 = energy_tmp
        #print(energy1)
        #print()
        return psi_new

    def calc_cnfig_weight(self, cnfig=None):
        """
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            cnfig is a 01 matrix
            a better abbreviation for configuration, rather than cnfig, is required
            if cnfig has been calculated, return that answer
            wait to complete
        """
        if cnfig==None:
            cnfig = self.spins
        if self.memory[to_bin(cnfig)] is not None:
            return self.memory[to_bin(cnfig)]
        n = len(cnfig)
        m = len(cnfig[0])
        ans = [self.redu_tensor[-1][i][cnfig[-1][i]] for i in range(m)]
        for i in range(n-2,0,-1):
            #print(i)
            ans = square_lattice.contract_two_row(ans, [self.redu_tensor[i][j][cnfig[i][j]] for j in range(m)])
        ans = very_simple_contract([[self.redu_tensor[0][i][cnfig[0][i]] for i in range(m)],ans], 2, m).tolist()
        self.memory[to_bin(cnfig)] = ans
        print("saving: ", to_bin(cnfig))
        return ans

    def calc_cnfig_energy(self, cnfig=None):
        if cnfig==None:
            cnfig = self.spins
        aux_s = [[0 for _ in range(self.m)] for _ in range(self.n)]
        w0 = self.calc_cnfig_weight()
        ans = 0
        E = 0
        for i in range(self.n):
            for j in range(self.m):
                if (i+1) in range(self.n):
                    E += self.H[cnfig[i][j]][cnfig[i+1][j]][aux_s[i][j]][aux_s[i+1][j]]
                if (j+1) in range(self.m):
                    E += self.H[cnfig[i][j]][cnfig[i][j+1]][aux_s[i][j]][aux_s[i][j+1]]
        i = 0
        while True:
            w1 = self.calc_cnfig_weight(aux_s)
            ans += E * w1 / w0
            print(aux_s, E, w0, w1, ans)
            i += 1
            pos = 0
            tmp = i
            while tmp % 2 == 0:
                tmp = tmp >> 1
                pos += 1
            x = pos // self.m
            y = pos % self.m
            if x not in range(self.n):
                break
            tmp = aux_s[x][y]
            for k in range(4):
                dx = (k % 2) * (2 - k)
                dy = ((k+1) % 2) * (1 - k)
                if (x + dx in range(self.n)) and (y + dy in range(self.m)):
                    E += self.H[cnfig[x][y]][cnfig[x+dx][y+dy]][(1-tmp)][aux_s[x+dx][y+dy]] \
                         - self.H[cnfig[x][y]][cnfig[x+dx][y+dy]][tmp][aux_s[x+dx][y+dy]]
            aux_s[x][y] = 1 - tmp
        return ans

    def evolve(self):
        new_spins = attempt_step(self.spins)
        w1 = self.calc_cnfig_weight() ** 2
        w2 = self.calc_cnfig_weight(new_spins) ** 2
        p = min(1, w2 / w1)
        if np.random.rand()<p:
            self.spins = new_spins

    def preheat(self, preheat_time):
        for i in range(preheat_time):
            print("preheat:", i)
            print("spins now is:", self.spins)
            self.evolve()

    def sampling(self, sampling_time):
        ans = 0
        for i in range(sampling_time):
            print("sampling:", i)
            self.evolve()
            ans += self.calc_cnfig_energy()
            print("Now energy is:", ans / (i+1) / 16)
        return ans / sampling_time / 16
