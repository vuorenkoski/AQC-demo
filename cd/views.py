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
from io import BytesIO
import base64

colors=['gray', 'blue','red','green','magenta','yellow','purple','black']
solvers=['local simulator', 'cloud hybrid solver', 'quantum solver']

def index(request):
    if request.method == "POST":
        size = int(request.POST['size'])
        seed = int(request.POST['seed'])
        communities = int(request.POST['communities'])
        max_weight = int(request.POST['max_weight'])
        num_reads = int(request.POST['reads'])
        solver = request.POST['solver']
        token = request.POST['token']

        if size<1 or size>150:
            return render(request, 'cd/index.html', {'seed':seed, 'vertices':size, 'communities':communities, 'max_weight':max_weight, 'token':token,
                                                     'solvers':solvers, 'solver':solver, 'reads':num_reads, 'error': 'vertices must be 1..150'}) 
        if communities<1 or communities>7:
            return render(request, 'cd/index.html', {'seed':seed, 'vertices':size, 'communities':communities, 'max_weight':max_weight, 'token':token,
                                                     'solvers':solvers, 'solver':solver, 'reads':num_reads, 'error': 'communities must be 1..7'}) 

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
        bqm = dimod.BinaryQuadraticModel(Q, 'BINARY').relabel_variables(labels, inplace=False) 
        result = {}
        result['edges'] = len(G.edges)
        result['vertices'] = len(G.nodes)
        result['qubo_size'] = Q.shape[0]
        result['logical_qubits'] = Q.shape[0]  
        result['couplers'] = len(bqm.quadratic)
        if solver=='local simulator':
            ts = time.time()
            sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads).aggregate()
            result['time'] = int((time.time()-ts)*1000)
            hist = print_histogram(sampleset, fig_size=5)
            result['occurences'] = int(sampleset.first.num_occurrences)
        elif solver=='cloud hybrid solver':
            try:
                sampleset = LeapHybridSampler(token=token).sample(bqm).aggregate()
            except Exception as err:
                return render(request, 'cd/index.html', {'seed':seed, 'vertices':size, 'communities':communities, 'max_weight':max_weight, 'token':token,
                                                     'solvers':solvers, 'solver':solver, 'reads':num_reads, 'error': err}) 
            result['time'] = int(sampleset.info['qpu_access_time'] / 1000)
            hist = None
        elif solver=='quantum solver':
            try:
                machine = DWaveSampler(token=token)
                result['chipset'] = machine.properties['chip_id']
                sampleset = EmbeddingComposite(machine).sample(bqm, num_reads=num_reads).aggregate()
            except Exception as err:
                return render(request, 'cd/index.html', {'seed':seed, 'vertices':size, 'communities':communities, 'max_weight':max_weight, 'token':token,
                                                     'solvers':solvers, 'solver':solver, 'reads':num_reads, 'error': err}) 
            result['time'] = int(sampleset.info['timing']['qpu_access_time'] / 1000)
            result['physical_qubits'] = sum(len(x) for x in sampleset.info['embedding_context']['embedding'].values())
            result['chainb'] = sampleset.first.chain_break_fraction
            hist = print_histogram(sampleset, fig_size=5)
            result['occurences'] = int(sampleset.first.num_occurrences)
        result['energy'] = int(sampleset.first.energy)
        nc = result_to_colors(G,sampleset.first.sample)
        graph = print_graph(G, node_color=nc, fig_size=5)
    else:
        solver = 'local simulator'
        token = ''
        size = 20
        seed = 42
        communities = 4
        max_weight = 10
        num_reads = 1000
        graph = None
        hist = None
        result = {}
    return render(request, 'cd/index.html', {'graph':graph, 'result':result, 'seed':seed, 'vertices':size, 'token':token,
                  'communities':communities, 'max_weight':max_weight, 'solvers':solvers, 'solver':solver, 'reads':num_reads, 'histogram':hist}) 

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

def print_histogram(sampleset, fig_size=6):
    data = {}
    maxv = int(sampleset.first.energy)
    minv = int(sampleset.first.energy)
    for e,n in sampleset.data(fields=['energy','num_occurrences']):
        energy = int(e)
        minv = min(energy,minv)
        maxv = max(energy,maxv)
        if energy in data.keys():
            data[energy] += n
        else:
            data[energy] = n
    labels = []
    datap = []
    for i in range(minv,maxv):
        labels.append(i)
        if i in data.keys():
            datap.append(data[i])
        else:
            datap.append(0)

    plt.switch_backend('AGG')
    plt.figure(figsize=(fig_size, fig_size))
    plt.bar(labels,datap)
    plt.xlabel('Energy')
    plt.ylabel('Occcurrences')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def result_to_colors(G, sample):
    cs = np.zeros(len(G.nodes))
    for k,v in sample.items():
        if v==1: 
            cs[k[0]]=k[1]+1
    nc = []
    for i in range(len(cs)):
        nc.append(colors[int(cs[i])])
    return nc