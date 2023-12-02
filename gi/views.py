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

solvers = ['local heuristic solver', 'cloud hybrid solver', 'quantum solver']
gtypes = ['identical', 'permuted', 'random']

def index(request):
    if request.method == "POST":
        size = int(request.POST['size'])
        seed = int(request.POST['seed'])
        gtype = request.POST['type']
        num_reads = int(request.POST['reads'])
        solver = request.POST['solver']
        token = request.POST['token']

        if size<1 or size>50:
            return render(request, 'gi/index.html', {'seed':seed, 'vertices':size, 'types':gtypes, 'type':gtype, 'token':token,
                                                     'solvers':solvers, 'solver':solver, 'reads':num_reads, 'error': 'vertices must be 1..50'}) 

        random.seed(seed)
        G1 = nx.random_geometric_graph(size, 0.5, seed=seed)
        E1 = [] 
        E2 = [] 
        for e in G1.edges(data=True):
            E1.append((e[0],e[1]))
        p=len(E1)
        if gtype=='identical':
            G2 = G1
            for e in G1.edges(data=True):
                E2.append((e[0],e[1]))
        elif gtype=='permuted':
            mapping = dict(zip(G1.nodes(), sorted(G1.nodes(), key=lambda k: random.random())))
            G2 = nx.relabel_nodes(G1, mapping)
            for e in G2.edges(data=True):
                E2.append((e[0],e[1]))
        elif gtype=='random':
            i = 1
            G2 = nx.random_geometric_graph(size, 0.5, seed=seed+i) 
            while len(G2.edges)!=len(G1.edges):
                i += 1
                G2 = nx.random_geometric_graph(size, 0.5, seed=seed+i) 
            for e in G2.edges(data=True):
                E2.append((e[0],e[1]))

        labels = {}
        for i in range(size):
            for j in range(size):
                labels[i*size+j] = (i,j)

        Q = create_qubo(E1, E2, size, p)
        bqm = dimod.BinaryQuadraticModel(Q, 'BINARY').relabel_variables(labels, inplace=False) 
        result = {}
        result['edges'] = len(G1.edges)
        result['vertices'] = len(G1.nodes)
        result['qubo_size'] = Q.shape[0]
        result['logical_qubits'] = Q.shape[0]
        result['exp_energy'] = -len(E1)
        result['couplers'] = len(bqm.quadratic)
        if solver=='local heuristic solver':
            ts = time.time()
            sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads).aggregate()
            result['time'] = int((time.time()-ts)*1000)
            hist = print_histogram(sampleset, fig_size=5)
            result['occurences'] = int(sampleset.first.num_occurrences)
        elif solver=='cloud hybrid solver':
            try:
                sampleset = LeapHybridSampler(token=token).sample(bqm).aggregate()
            except Exception as err:
                return render(request, 'gi/index.html', {'seed':seed, 'vertices':size, 'types':gtypes, 'type':gtype, 'token':token,
                                                     'solvers':solvers, 'solver':solver, 'reads':num_reads, 'error': err}) 
            result['time'] = int(sampleset.info['qpu_access_time'] / 1000)
            hist = None
        elif solver=='quantum solver':
            try:
                machine = DWaveSampler(token=token)
                result['chipset'] = machine.properties['chip_id']
                sampleset = EmbeddingComposite(machine).sample(bqm, num_reads=num_reads).aggregate()
            except Exception as err:
                return render(request, 'gi/index.html', {'seed':seed, 'vertices':size, 'types':gtypes, 'type':gtype, 'token':token,
                                                     'solvers':solvers, 'solver':solver, 'reads':num_reads, 'error': err}) 
            result['time'] = int(sampleset.info['timing']['qpu_access_time'] / 1000)
            result['physical_qubits'] = sum(len(x) for x in sampleset.info['embedding_context']['embedding'].values())
            result['chainb'] = sampleset.first.chain_break_fraction
            hist = print_histogram(sampleset, fig_size=5)
            result['occurences'] = int(sampleset.first.num_occurrences)
        result['energy'] = int(sampleset.first.energy)
        print(solver)
        if result['exp_energy']==result['energy']:
            result['result']='isomorphic'
        else:
            result['result']='non-isomorphic'
        graph1 = print_graph(G1, fig_size=4)
        graph2 = print_graph(G2, fig_size=4)
    else:
        solver = 'local heuristic solver'
        gtype = 'identical'
        token = ''
        size = 7
        seed = 42
        num_reads = 1000
        graph1 = None
        graph2 = None
        hist = None
        result = {}
    return render(request, 'gi/index.html', {'graph1':graph1, 'graph2':graph2, 'result':result, 'seed':seed, 'vertices':size, 'token':token,
                  'type':gtype, 'types':gtypes, 'solvers':solvers, 'solver':solver, 'reads':num_reads, 'histogram':hist}) 

def create_qubo(E1,E2,vertices,p):
    Q = np.zeros((vertices*vertices, vertices*vertices))
    
    # Constraint 1: penalty if several mappings from same source
    for i in range(vertices): 
        for j in range(vertices): 
            for k in range(j+1,vertices): 
                Q[i*vertices+j,i*vertices+k]=p 

    # Constaint 2: penalty if several mappings to same target
    for i in range(vertices): 
        for j in range(vertices): 
            for k in range(j+1,vertices): 
                Q[i+vertices*j,i+vertices*k]=p 
                
    # Constraint 3: -1 for each succesfully mapped edge: (x1,y1) -> (x2,y2) 
    #    two possible mappings: (x1->x2, y1->y2) or (x1->y2,y1->x2)
    for e1 in E1: 
        for e2 in E2: 
            Q[e1[0]*vertices+e2[0], e1[1]*vertices+e2[1]] -= 1
            Q[e1[0]*vertices+e2[1], e1[1]*vertices+e2[0]] -= 1
            
    # All quadratic coefficients in lower triangle to upper triangle
    for i in range(vertices): 
        for j in range(i):
            Q[j,i] += Q[i,j]
            Q[i,j] = 0
    return Q

def print_graph(G, fig_size=6):
    plt.switch_backend('AGG')
    pos = nx.spring_layout(G)
    plt.figure(figsize=(fig_size, fig_size))
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_nodes(G, pos, node_size=80)
    plt.axis("off")
    plt.show()
    
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
