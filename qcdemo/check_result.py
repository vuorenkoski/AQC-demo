from networkx.classes.function import path_weight
from networkx import has_path, all_shortest_paths
import networkx as nx

def xy_from_label(e):
    x = 0
    y = 0
    i = len(e)-1
    d = 1
    while e[i]!='-':
        y += d*int(e[i])
        d *= 10 
        i -= 1
    d = 1
    i -= 1
    while i>=0:
        x += d*int(e[i])
        d *= 10
        i -= 1 
    return (x,y)

def path_from_sample(sample,G):
    vertices = len(G.nodes)
    s = 0
    si = 0
    t = 0
    ti = 0
    w = 0
    for v in range(vertices):
        if sample['s'+str(v)]==1:
            s = v
            si += 1
        if sample['t'+str(v)]==1:
            t = v
            ti += 1
    if s==t or si!=1 or ti!=1: # only one s and t allowed, not the same
        return (None,None,None)
    i = s
    path = [i]
    vv = 1
    while i!=t:
        xx = i
        for p in sample:
            if p[0]!='s' and p[0]!='t' and sample[p]==1:
                x,y = xy_from_label(p)
                if x==i:
                    path.append(y)
                    w += G[x][y]['weight']
                    i = y
                    break
        vv += 1
        if vv>vertices:
            return (None,None,None)
    return (str(s)+'-'+str(t),path,w)

def result_paths(G, sampleset): # return all legal shrotest paths from sampleset
    res = {}
    for s in sampleset.filter(lambda s: s.energy<0):
        st, path, w = path_from_sample(s,G)
        if st!=None:
            if st not in res:
                res[st]=(path,w)
            else:
                if res[st][1]>w:
                    res[st]=(path,w)
    return res

def check_result_apsp(G,sampleset):
    ok = 0
    s = 0
    res = result_paths(G, sampleset)
    for i in G.nodes:
        for j in G.nodes:
            if i!=j and has_path(G,i,j):
                s += 1
                sp = [p for p in all_shortest_paths(G,i,j,weight='weight')] # correct shorstest paths for this pair
                w1 = path_weight(G,sp[0],'weight') # correct weight
                if str(i)+'-'+str(j) in res.keys():  # Does result have path between s-t
                    path,w2 = res[str(i)+'-'+str(j)]
                    if (path in sp) and w1==w2:      # Is path among correct paths and are weights same
                        ok += 1
    return str(int(100*ok/s))+'%'

def check_result_gi(sampleset, e):
    if int(sampleset.first.energy)==e:
        return 'isomorphic'
    else:
        return 'non-isomorphic'

def check_result_cd(G, sampleset, communities):
    c1 = nx.community.greedy_modularity_communities(G, weight='weight', best_n=communities)
    res1 = nx.community.modularity(G,c1)
    c2 = [[] for i in range(communities)]
    for k,v in sampleset.first.sample.items():
        if v==1: 
            c2[k[1]].append(k[0])
    c2 = [frozenset(x) for x in c2]

    if nx.community.is_partition(G,c2):
        res2 = nx.community.modularity(G,c2)
        return str(round(res2-res1,3))
    else:
        return 'np'
