from django.shortcuts import render
from django.http import HttpResponse

import matplotlib.pyplot as plt
import numpy as np
import time
import random
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler
import dwave.inspector
import networkx as nx
from networkx.classes.function import path_weight
from io import BytesIO
import base64

colors=['gray', 'blue','red','green','magenta','yellow','purple','black']
solvers=['local heuristic solver', 'cloud hybrid solver']

def index(request):
    fig_size = 6
    if request.method == "POST":
        size = int(request.POST['size'])
        seed = int(request.POST['seed'])
        communities = int(request.POST['communities'])
        max_weight = int(request.POST['max_weight'])
        solver = request.POST['solver']

        random.seed(seed)
        G = nx.random_geometric_graph(size, 0.2, seed=seed)
        nx.set_edge_attributes(G, {e: {'weight': random.randint(1, max_weight)} for e in G.edges})

        max_count = 0
        for e in G.edges:
            max_count += G[e[0]][e[1]]['weight']

        labels = {}
        for i in range(len(G.nodes)):
            for j in range(communities):
                labels[i*communities + j]=(i,j)
                
        Q = create_qubo(G, communities, max_count)
        bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')
        bqm = bqm.relabel_variables(labels, inplace=False) 

        num_reads = 1000
        ts = time.time()
        sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads)
        energy = int(sampleset.first.energy)
        det_time = int((time.time()-ts)*1000)
        nc = result_to_colors(G,sampleset.first.sample)
        graph = print_graph(G, node_color=nc, fig_size=fig_size)
    else:
        solver = 'local heuristic solver'
        size = 20
        seed = 42
        communities = 4
        max_weight = 10
        graph = None
        det_time = None
        energy = None
    return render(request, 'cd/index.html', {'graph':graph, 'time':det_time, 'seed':seed, 'vertices':size, 
                  'communities':communities, 'max_weight':max_weight, 'energy':energy, 'solvers':solvers, 'solver':solver}) 

def create_qubo(G, communities, p):
    vertices = len(G.nodes)
    Q = np.zeros((vertices*communities, vertices*communities))
    
    # Helper datastructure to containt k
    k = np.zeros(vertices)
    for e in G.edges:
        k[e[0]] += G[e[0]][e[1]]['weight']
        k[e[1]] += G[e[0]][e[1]]['weight']

    # Constraint 1
    for v in range(vertices): 
        for c1 in range(communities): 
            for c2 in range(communities):
                if  c1!=c2:
                    Q[v*communities+c1,v*communities+c2] += p
                
    # Constraint 2
    for c in range(communities):
        for v1 in range(vertices): 
            for v2 in range(v1+1,vertices): 
                Q[v1*communities+c, v2*communities+c] += k[v1]*k[v2] / (2*p)
                
    for e in G.edges:
        for c in range(communities):
            Q[e[0]*communities+c, e[1]*communities+c] -= G[e[0]][e[1]]['weight']
            
    return Q

def print_graph(G, pos=None, node_color=None, fig_size=6):
    if pos==None:
        pos = nx.get_node_attributes(G, 'pos')
    m = 0
    for k,v in nx.get_edge_attributes(G, 'weight').items():
        m = max(m,v)
    a = [v/m for k,v in nx.get_edge_attributes(G, 'weight').items()]

    plt.switch_backend('AGG')
    plt.figure(figsize=(fig_size, fig_size))
    nx.draw_networkx_edges(G, pos, alpha=a)
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color=node_color)
    plt.axis("off")
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

colors=['gray', 'blue','red','green','magenta','yellow','purple','black']

def result_to_colors(G, sample):
    cs = np.zeros(len(G.nodes))
    for k,v in sample.items():
        if v==1: 
            cs[k[0]]=k[1]+1
    nc = []
    for i in range(len(cs)):
        nc.append(colors[int(cs[i])])
    return nc

def solve_random_graph(seed, size=50, communities=4, fig_size=6):
    max_weight = 10
    random.seed(seed)
    G = nx.random_geometric_graph(size, 0.3, seed=seed)
    nx.set_edge_attributes(G, {e: {'weight': random.randint(1, max_weight)} for e in G.edges})
    
    max_count = 0
    for e in G.edges:
        max_count += G[e[0]][e[1]]['weight']

    labels = {}
    for i in range(len(G.nodes)):
        for j in range(communities):
            labels[i*communities + j]=(i,j)
            
    Q = create_qubo(G, communities, max_count)
    bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')
    bqm = bqm.relabel_variables(labels, inplace=False) 
    
    print('\nNumber of logical qubits needed:',Q.shape[0])
    print('Number of couplers needed:', len(bqm.quadratic))
    print('\nSimulator solver')
    num_reads = 1000
    ts = time.time()
    sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads)
    det_time = (time.time()-ts)*1000
    print('Time used (ms): {:.3f}\n'.format(det_time))
    nc = result_to_colors(G,sampleset.first.sample)
    print_graph(G, node_color=nc, fig_size=fig_size)
    
    print('Hybrid solver')
    sampleset = LeapHybridSampler().sample(bqm)
    hyb_time = sampleset.info['qpu_access_time'] / 1000
    run_time = sampleset.info['run_time'] / 1000
    print('QPU time used (ms): {:.1f}'.format(hyb_time))
    print('Total time used (ms): {:.1f}\n'.format(run_time))
    nc = result_to_colors(G,sampleset.first.sample)
    print_graph(G, node_color=nc)