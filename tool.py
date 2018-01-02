# -*- coding: utf-8 -*-

"""
    put some relevent functions here
    these things should be handled more seriously...
"""

def decompose_tool(func, T, tag, tag1, tag2):
    tmp_tag = T.tags
    tmp = tmp_tag.index(tag)
    T.transpose(tmp_tag[:tmp] + tmp[tmp+1:] + [tag])
    q, r = func(T, len(T.tags)-1, tag1, tag2)
    T.transpose(tmp_tag)

def very_simple_contract(sq_latt, n, m, u="u", d="d", l="l", r="r"):
    #   u
    # l   r
    #   d
    ans = Node([], [])
    for j in range(m):
        for i in range(n):
            tag1 = []
            tag2 = []
            if i!=0:
                tag1 += [d]
                tag2 += [u]
            if j!=0:
                tag1 += [str(i)]
                tag2 += [l]
            if j!=(m-1):
                dict2 = {}
            else:
                dict2 = {r:str(j)}
            ans = Node.contract(ans, tag1, sq_latt[i][j], tag2, {}, dict2)
    return np.copy(ans.data)

def unitarilize(node_list, target_tag, counter_tag):
    for i in range(len(node_list)-1):
        node_list[i] , r = \
            decompose_tool(Node.qr, node_list[i], target_tag, target_tag, counter_tag)
        node_list[i+1] = \
            Node.contract(node_list[i+1], [counter_tag], r, [target_tag])

