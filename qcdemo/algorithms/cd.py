from django.shortcuts import render

import numpy as np
from dimod import BinaryQuadraticModel
import networkx as nx
import numpy as np

from qcdemo.graphs import create_graph
from qcdemo.utils import basic_stats, solve, graph_to_json, Q_to_json, colors, algorithms, graph_types

min_vertices = 5
max_vertices = 20
min_communities = 1
max_communities = 10
max_num_reads = 10000
solvers = ['local simulator', 'quantum solver', 'cloud hybrid solver']

def index(request):
    resp = {}
    resp['algorithm'] = 'Community detection'
    resp['correctness'] = 'Community graphs have three artificial communities. Correctness is measured by the difference of modularity value of the'\
         'outcome and the modularity value given of outcome of NetworkX function greedy_modularity_communities. If difference is 0.0, outcomes are'\
         'the same. More negative the value is, more poorer the modularity of the algorithms outcome was.'
    resp['algorithms'] = algorithms
    resp['solvers'] = solvers
    resp['graph_types'] = graph_types
    resp['min_vertices'] = min_vertices
    resp['max_vertices'] = max_vertices
    resp['min_communities'] = min_communities
    resp['max_communities'] = max_communities
    resp['max_num_reads'] = max_num_reads
    if request.method == "POST":
        # Get parameters
        resp['vertices'] = int(request.POST['vertices'])
        resp['num_reads'] = int(request.POST['num_reads'])
        resp['solver'] = request.POST['solver']
        resp['token'] = request.POST['token']
        resp['graph_type'] = request.POST['graph_type']
        resp['communities'] = int(request.POST['communities'])

        # Check validity
        if resp['vertices']<min_vertices or resp['vertices']>max_vertices:
            resp['error'] = 'vertices must be '+str(min_vertices)+'..'+str(max_vertices)
            return render(request, 'cd/index.html', resp) 

        if resp['communities']<min_communities or resp['communities']>max_communities:
            resp['error'] = 'communities must be '+str(min_communities)+'..'+str(max_communities)
            return render(request, 'cd/index.html', resp) 

        if resp['num_reads']>max_num_reads:
            resp['error'] = 'Maximum number fo reads is '+str(max_num_reads)
            return render(request, 'cd/index.html', resp) 

        # create graph, qubo, bqm
        G = create_graph(resp['graph_type'],resp['vertices'], weight=True, directed=False)
        Q = create_qubo_cd(G, resp['communities'])
        bqm = create_bqm_gi(Q, G, resp['communities'])
        result = basic_stats(G,Q, bqm)
        resp['qdata'] = {'data': Q_to_json(Q.tolist()), 'size':len(Q)}

        # Solve
        try:
            r, sampleset = solve(bqm,resp)
            result.update(r)
        except Exception as err:
            resp['error'] = err
            return render(request, 'algorithm.html', resp) 

        # Gather rest of results
        result['energy'] = int(sampleset.first.energy)
        result['success'] = check_result_cd(G,sampleset,resp['communities'])
        resp['result'] = result

        # Create graph-data
        resp['gdata'] = {'data': graph_to_json(G), 'colors': result_to_colors(G, sampleset.first.sample), 'directed':0, 'weights':1}
    else:
        # These are initial parameters for web page
        resp['vertices'] = 7
        resp['num_reads'] = 2000
        resp['solver'] = 'local simulator'
        resp['token'] = ''
        resp['graph_type'] = 'community graph'
        resp['communities'] = 4
    return render(request, 'algorithm.html', resp) 

def create_bqm_gi(Q, G, communities):
    labels = {}
    for i in range(len(G.nodes)):
        for j in range(communities):
            labels[i*communities + j]=(i,j)
    return BinaryQuadraticModel(Q, 'BINARY').relabel_variables(labels, inplace=False)

def result_to_colors(G, sample):
    cs = np.zeros(len(G.nodes))
    for k,v in sample.items():
        if v==1: 
            cs[k[0]]=k[1]+1
    nc = []
    for i in range(len(cs)):
        nc.append(colors[int(cs[i])])
    return nc

def create_qubo_cd(G, communities):
    vertices = len(G.nodes)
    Q = np.zeros((vertices*communities, vertices*communities))
    p = 0
    for e in G.edges:
        p += G[e[0]][e[1]]['weight']

    # Helper datastructure to containt k
    k = np.zeros(vertices)
    for e in G.edges:
        k[e[0]] += G[e[0]][e[1]]['weight']
        k[e[1]] += G[e[0]][e[1]]['weight']

    # Constraint 1
    for v in range(vertices): 
        for c1 in range(communities):
            Q[v*communities+c1,v*communities+c1] -= p
            for c2 in range(c1+1, communities):
                Q[v*communities+c1,v*communities+c2] += 2*p
                
    # Constraint 2
    for c in range(communities):
        for v1 in range(vertices): 
            for v2 in range(v1+1,vertices): 
                Q[v1*communities+c, v2*communities+c] += k[v1]*k[v2] / (2*p)
                
    for e in G.edges:
        for c in range(communities):
            Q[e[0]*communities+c, e[1]*communities+c] -= G[e[0]][e[1]]['weight']
            
    return Q

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