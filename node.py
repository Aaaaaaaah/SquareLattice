# -*- coding: utf-8 -*-

import numpy as np

var_pool = []
prog_pool = []
modules = ["tools","tensor_type"]
program_name = "main"

def output_program(file):
    global var_pool
    global prog_pool
    file.write("program %s\n"%program_name)
    for i in modules:
        file.write("use %s\n"%i)
    file.write("implicit none\n")
    for i in var_pool:
        file.write("type(tensor):: %s\n"%i)
    for i in prog_pool:
        file.write("%s\n"%i)
    file.write("end program %s\n"%program_name)

class TnspVar:
    def __init__(self,shape):
        global var_pool
        self.data = "TNS%d"%len(var_pool)
        var_pool.append(self.data)
        prog_pool.append("call%s%%allocate(%s,'real')"%(self.data,str(list(shape))))
    def write(self):
        global prog_pool
        prog_pool.append("call %s%%write(6)"%self.data)
    def fill_data(self,data):
        """
        data maybe python array or other TnspVar
        """
        if isinstance(data,TnspVar):
            #!!!
        else:
            #!!!!
            pass
    def random_init(self):
        """
        init data in the tensor with uniform
        """
        global prog_pool
        prog.pool.append("call %s%%random()"%self.data)
    def ones_init(self):
        """
        fill ones in the tensor
        """
        global prog_pool
        prog.pool.append("call %s%%setValue(1)"%self.data)
    def div_max(self):
        global prog_pool
        prog_pool.append("%s=%s/(%s%%dmaxmin(maxabs))"%(self.data,self.data,self.data))

class Node(TnspVar):

    def __init__(self,tags,dims,data=None,envs=None):
        assert len(tags) == len(dims)
        assert len(set(tags)) == len(tags)
        if data is not None:
            super(Node,self).__init__(dims)
            self.fill_data(data)
            self.div_max()
        else:
            super(Node,self).__init__(dims)
            self.random_init()
        if envs is not None:
            self.envs = []
            for i,j in zip(dims,envs):
                if j is None:
                    mp = TnspVar([i,])
                    tmp.ones_init()
                    self.envs.append(tmp)
                else:
                    tmp = TnspVar([i,])
                    tmp.fill_data(j)
                    tmp.div_max()
                    assert tmp.shape == (i,)
                    self.envs.append(tmp)
        else:
            self.envs = []
            for i in dims:
                tmp = TnspVar([i,])
                tmp.ones()
                self.envs.append(tmp)
        self.dims = list(dims)
        self.tags = list(tags)

    def replace(self,other):
        self.data = other.data#!!!
        self.envs = other.envs
        self.dims = other.dims
        self.tags = other.tags

    def rename_leg(self,tag_dict):
        for i,j in tag_dict.items():
            self.tags[self.tags.index(i)] = j

    @staticmethod
    def absorb_envs(self,pow,legs=None):
        ans = self
        if legs == None:
            legs = range(len(self.dims))
        for i in legs:
            tmp = np.ones(len(self.dims),dtype=int)
            tmp[i] = self.dims[i]
            temp = tf.reshape(self.envs[i],tmp) #!!!
            if pow > 0:
                for _ in range(pow):
                    ans *= temp #!!!!
            else:
                for _ in range(-pow):
                    ans /= temp #!!!
        return ans

    def transpose(self,tags):
        self.data = np.transpose(self.data,[self.tags.index(i) for i in tags]) #!!!
        tmp = self.dims
        self.dims = [tmp[self.tags.index(i)] for i in tags]
        tmp = self.envs
        self.envs = [tmp[self.tags.index(i)] for i in tags]
        self.tags = tags

    def __repr__(self):
        return "Node with dims: %s"%str(zip(self.tags,self.dims))

    @staticmethod
    def contract(T1,tags1,T2,tags2,tags_dict1=dict(),tags_dict2=dict()):
        # order:the indexs of legs waiting for contracting
        order1 = [T1.tags.index(i) for i in tags1]
        order2 = [T2.tags.index(i) for i in tags2]
        # absorb envsironment
        TD1 = Node.absorb_envs(T1,1,order1)
        TD2 = Node.absorb_envs(T2,1,order2)
        # generate the contribute of the answer
        for i in tags_dict1:
            assert i in T1.tags and i is not tags1
        for i in tags_dict2:
            assert i in T2.tags and i is not tags2
        tags = [j if j not in tags_dict1 else tags_dict1[j] for i,j in enumerate(T1.tags) if i not in order1] +\
               [j if j not in tags_dict2 else tags_dict2[j] for i,j in enumerate(T2.tags) if i not in order2]
        dims = [j for i,j in enumerate(T1.dims) if i not in order1] + [j for i,j in enumerate(T2.dims) if i not in order2]
        envs = [j for i,j in enumerate(T1.envs) if i not in order1] + [j for i,j in enumerate(T2.envs) if i not in order2]
        #initiate the answer
        T = Node(tags,dims,np.tensordot(TD1,TD2,[order1,order2]),envs) #!!!
        return T

    @staticmethod
    def svd(self,num,tag1,tag2,cut):
        dims1 = self.dims[:num]
        dims2 = self.dims[num:]
        data1, env, data2 = np.linalg.svd( #!!!
            np.reshape(
                Node.absorb_envs(self,2),
                [np.prod(dims1),np.prod(dims2)])
        )
        env = np.sqrt(env[:cut]) #!!!
        data1 = data1[:,:cut]
        data2 = data2[:cut,:]
        tags1 = self.tags[:num] + [tag1]
        tags2 = [tag2] + self.tags[num:]
        dims1 = dims1 + [cut]
        dims2 = [cut] + dims2
        envs1 = self.envs[:num] + [env]
        envs2 = [env] + self.envs[num:]
        T1,T2 = Node(tags1,dims1,data1,envs1),Node(tags2,dims2,data2,envs2)
        T1.data = Node.absorb_envs(T1,-2,range(len(dims1)-1))
        T2.data = Node.absorb_envs(T2,-2,range(1,len(dims2)))
        return T1,T2

    @staticmethod
    def update(T1,T2,tag1,tag2,phy1,phy2,H,cut=None):
        # 准备
        l1 = T1.dims[T1.tags.index(phy1)]
        l2 = T2.dims[T2.tags.index(phy2)]
        if cut is None:
            cut = T1.dims[T1.tags.index(tag1)]

        # 缩并
        TD = Node.contract(T1,[tag1],T2,[tag2],
                           {i:"__1.%s"%i for i in T1.tags if i is not tag1},
                           {i:"__2.%s"%i for i in T2.tags if i is not tag2})
        tmp = TD.tags
        HH = Node(["__1","__2","__1.%s"%phy1,"__2.%s"%phy2],[l1,l2,l1,l2],H)
        TD = Node.contract(TD,["__1.%s"%phy1,"__2.%s"%phy2],HH,["__1","__2"])
        TD.transpose(tmp)
        # SVD
        TD1,TD2 = Node.svd(TD,len(T1.tags)-1,tag1,tag2,cut)
        TD1.rename_leg({"__1.%s"%i:i for i in T1.tags if i is not tag1})
        TD2.rename_leg({"__2.%s"%i:i for i in T2.tags if i is not tag2})
        TD1.transpose(T1.tags)
        TD2.transpose(T2.tags)
        T1.replace(TD1) #!!!
        T2.replace(TD2) #!!!

