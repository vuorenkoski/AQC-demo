# Path graph: graph that can be drawn so that all of its vertices and edges lie on a single straight line
# Star graph: graph that gave n-1 edges from singkle node to all other nodes
# Cycle graph: graph in which all nodes form single cycle.
# Complete graph: graph having all possible edges
# Tree graph: graph in which any two vertices are connected by exactly one path
# Graph with single cycle: graph in which part of the nodes form single cycle.
# Graph with multiple cycles: graph having multiple cycles.
# Bipartite graph: graph which vertices can be divided in two sets so that all edges are between sets
# Wheel graph: graph formed by connecting one vertex to all other vertices which together forms a cycle.

import networkx as nx
import random

graph_types = ['path graph', 'star graph', 'cycle graph', 'complete graph', 'tree graph', 'single cycle graph', 
               'multiple cycle graph', 'bipartite graph', 'regular graph', 'wheel graph', 'friendship graph',
               'random graph']
    
def create_graph(name, vertices, structure, weight=False, directed=True, permutation=False):
    if name=='path graph':
        E = graph_path(vertices)
    elif name=='star graph':
        E = graph_star(vertices)
    elif name=='cycle graph':
        E = graph_cycle(vertices)
    elif name=='complete graph':
        E = graph_complete(vertices)
    elif name=='tree graph':
        E = graph_tree(vertices)
    elif name=='single cycle graph':
        E = graph_single_cycle(vertices)
    elif name=='multiple cycle graph':
        E = graph_multiple_cycle(vertices)
    elif name=='bipartite graph':
        E = graph_bipartite(vertices)
    elif name=='regular graph':
        E = graph_regular(vertices)
    elif name=='wheel graph':
        E = graph_wheel(vertices)
    elif name=='friendship graph':
        E = graph_friendship(vertices)
    elif name=='random graph':
        G = nx.gnp_random_graph(vertices, 0.30, seed=42, directed=directed)
        if weight:
            random.seed(42)
            nx.set_edge_attributes(G, {e: {'weight': random.randint(1, 10)} for e in G.edges})
        if permutation:
            mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
            GP = nx.relabel_nodes(G, mapping)
            return (G, GP)
        else:
            return G
    elif name=='community graph':
            return graph_community(vertices,3)
    elif name=='manual':
        lines = structure.split(', ')
        create_using = nx.DiGraph() if directed else nx.Graph()
        G = nx.parse_edgelist(lines, nodetype=int, data=(("weight", int),), create_using=create_using)
        if permutation:
            mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
            GP = nx.relabel_nodes(G, mapping)
            return (G, GP)
        else:
            return G
    else:
        E = []
    if E==[]:
        if permutation:
            return (None, None)
        else:
            return None
    
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(list(range(vertices)))
    for e in E:
        G.add_edge(e[0], e[1])
    if weight:
        random.seed(42)
        nx.set_edge_attributes(G, {e: {'weight': random.randint(1, 10)} for e in G.edges})
    if permutation:
        mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
        GP = nx.relabel_nodes(G, mapping)
        return (G, GP)
    else:
        return G

def graph_path(vertices):
    E = []
    if vertices<2:
        return E
    for i in range(vertices-1):
        E.append((i,i+1))
    return E

def graph_star(vertices):
    E = []
    if vertices<2:
        return E
    for i in range(vertices-1):
        E.append((0,i+1))
    return E

def graph_cycle(vertices):
    if vertices<3:
        return []
    E = graph_path(vertices)
    E.append((vertices-1,0))
    return E

def graph_complete(vertices):
    E = []
    if vertices<2:
        return E
    for i in range(vertices-1):
        for j in range(i+1,vertices):
            E.append((i,j))
    return E

def graph_tree(vertices):
    # balanced binary tree
    E = []
    if vertices<2:
        return E
    j = 1
    k = 0
    for i in range(vertices):
        if j+i*2<vertices:
            E.append((i,j+i*2))
        if j+i*2+1<vertices:
            E.append((i,j+i*2+1))
        k += 1
        if k==j:
            k = 0
            j *= j
    return E

def graph_single_cycle(vertices):
    # tree graph, but edge from last vertex to first vertex
    if vertices<4:
        return []
    E = graph_tree(vertices)
    E.append((vertices-1,0))
    return E

def graph_multiple_cycle(vertices):
    # tree graph, but edge from every second last level leaf to first vertex
    if vertices<5:
        return []
    E = graph_tree(vertices)
    j = 1
    k = 0
    while k+j<vertices:
        k += j 
        j *= 2
    for i in range(k,vertices,2):
        E.append((i,0))
    if vertices==5:
        E.append((4,0))
    return E

def graph_bipartite(vertices):
    E = []
    if vertices<2:
        return []
    for i in range(int((vertices+1)/2)):
        if i*2+3<vertices:
            E.append((i*2,i*2+3))
        if i*2+1<vertices:
            E.append((i*2,i*2+1))
        if i*2-1>0:
            E.append((i*2,i*2-1))
        if i*2-3>0:
            E.append((i*2,i*2-3))
    return E

def graph_regular(vertices):
    return []

def graph_wheel(vertices):
    if vertices<4:
        return []
    E = graph_cycle(vertices-1)
    for i in range(vertices-1):
        E.append((vertices-1,i))
    return E

def graph_friendship(vertices):
    return []

def graph_community(vertices,communities): 
    G = nx.Graph()
    vpc = int(vertices/communities)
    lo = vertices-vpc*communities
    index = 0
    c0 = []
    G.add_nodes_from([x for x in range(vertices)])
    for i in range(communities):
        if lo>0:
            c = vpc+1
            lo -= 1
        else:
            c = vpc
        for j in range(c-1):
            for k in range(j+1,c):
                G.add_weighted_edges_from([(index + j,index + k, 5)])
        c0.append(index)
        index += c
    for i in range(communities-1):
        for j in range(i+1,communities):
            G.add_weighted_edges_from([(c0[i],c0[j], 1)])
            G.add_weighted_edges_from([(c0[i]+1,c0[j]+1, 1)])
    return G
