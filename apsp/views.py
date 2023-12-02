from django.shortcuts import render
from django.http import HttpResponse

import matplotlib.pyplot as plt
import numpy as np
import time
import random
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler
import networkx as nx
from io import BytesIO
import base64
from networkx.classes.function import path_weight

solvers = ['local heuristic solver', 'quantum solver']

def index(request):
    if request.method == "POST":
        size = int(request.POST['size'])
        seed = int(request.POST['seed'])
        max_weight = int(request.POST['max_weight'])
        num_reads = int(request.POST['reads'])
        solver = request.POST['solver']
        token = request.POST['token']

        if size<1 or size>20:
            return render(request, 'apsp/index.html', {'seed':seed, 'vertices':size, 'max_weight':max_weight, 'token':token,
                                                     'solvers':solvers, 'solver':solver, 'reads':num_reads, 'error': 'vertices must be 1..20'}) 

        random.seed(seed)
        G = nx.gnp_random_graph(size, 0.30, seed, directed=True)
        nx.set_edge_attributes(G, {e: {'weight': random.randint(1, max_weight)} for e in G.edges})
        E = [] 
        for e in G.edges(data=True):
            E.append((e[0],e[1],e[2]['weight']))

        max_count = 0
        for e in E:
            max_count += e[2]

        labels = {}
        for i in range(size):
            labels[i]='s'+str(i)
            labels[size+i]='t'+str(i)   
        for i,e in enumerate(E):
            labels[size*2+i] = str(e[0]) + '-' + str(e[1])
            
        
        Q = create_qubo(E,size,max_count)
        bqm = dimod.BinaryQuadraticModel(Q, 'BINARY').relabel_variables(labels, inplace=False)

        result = {}
        result['edges'] = len(G.edges)
        result['vertices'] = len(G.nodes)
        result['qubo_size'] = Q.shape[0]
        result['logical_qubits'] = Q.shape[0]
        result['couplers'] = len(bqm.quadratic)
        if solver=='local heuristic solver':
            ts = time.time()
            sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads).aggregate()
            result['time'] = int((time.time()-ts)*1000)
            hist = print_histogram(sampleset, fig_size=5)
        elif solver=='cloud hybrid solver':
            try:
                sampleset = LeapHybridSampler(token=token).sample(bqm).aggregate()
            except Exception as err:
                return render(request, 'apsp/index.html', {'seed':seed, 'vertices':size, 'max_weight':max_weight, 'token':token,
                                                     'solvers':solvers, 'solver':solver, 'reads':num_reads, 'error': err}) 
            result['time'] = int(sampleset.info['qpu_access_time'] / 1000)
            hist = None
        elif solver=='quantum solver':
            try:
                machine = DWaveSampler(token=token)
                result['chipset'] = machine.properties['chip_id']
                sampleset = EmbeddingComposite(machine).sample(bqm, num_reads=num_reads).aggregate()
            except Exception as err:
                return render(request, 'apsp/index.html', {'seed':seed, 'vertices':size, 'max_weight':max_weight, 'token':token,
                                                     'solvers':solvers, 'solver':solver, 'reads':num_reads, 'error': err}) 
            result['time'] = int(sampleset.info['timing']['qpu_access_time'] / 1000)
            result['physical_qubits'] = sum(len(x) for x in sampleset.info['embedding_context']['embedding'].values())
            result['chainb'] = sampleset.first.chain_break_fraction
            hist = print_histogram(sampleset, fig_size=5)
        result['energy'] = int(sampleset.first.energy)
        graph = print_graph(G, fig_size=5)
        res, result['success'] = check_result(G,sampleset,E,size)
        result['paths']=[]
        for k,v in res.items():
            result['paths'].append({'nodes':k, 'path':str(v[0]), 'weight':v[1]})
    else:
        solver = 'local heuristic solver'
        token = ''
        size = 7
        seed = 42
        max_weight = 10
        num_reads = 2000
        graph = None
        hist = None
        result = {}
    return render(request, 'apsp/index.html', {'graph':graph, 'max_weight':max_weight, 'result':result, 'seed':seed, 'vertices':size, 'token':token,
                  'solvers':solvers, 'solver':solver, 'reads':num_reads, 'histogram':hist}) 

def create_qubo(E,vertices,p):
    edges = len(E)
    Q = np.zeros((2*vertices + edges, 2*vertices + edges))

    # Constraints 1 and 2
    for i in range(vertices):
        for j in range(vertices):
            if i!=j:
                Q[i,j] += p
                Q[vertices+i,j+vertices] += p
        
    # Constraint 3
    for i in range(vertices):
        Q[i,i+vertices] += p

    # Constraint 4
    for v in range(vertices):
        for i,e in enumerate(E):
            if e[0]==v:
                Q[v,vertices*2+i] -= p
            if e[1]==v:
                Q[v,vertices*2+i] += p

    # Constraint 5
    for v in range(vertices):
        for i,e in enumerate(E):
            if e[1]==v:
                Q[vertices+v,vertices*2+i] -= p
            if e[0]==v:
                Q[vertices+v,vertices*2+i] += p

    # Constraint 6 and 7
    for i,ei in enumerate(E):
        for j,ej in enumerate(E):
            if ei[0]==ej[0] or ei[1]==ej[1]:
                Q[vertices*2+i,vertices*2+j] += p
            if ei[1]==ej[0] or ei[0]==ej[1]:
                Q[vertices*2+i,vertices*2+j] -= p/2

    # Constraint 8 
    for i in range(edges):
        Q[vertices*2+i,vertices*2+i] += E[i][2]

    # Quadratic coefficients in lower triangle to upper triangle
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
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, font_color='white')
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
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

def path_from_sample(sample,E,vertices):
    s = 0
    t = 0
    w = 0
    for v in range(vertices):
        if sample['s'+str(v)]==1:
            s = v
        if sample['t'+str(v)]==1:
            t = v
    i = s
    path = [i]
    while i!=t:
        for e in E:
            if e[0]==i and sample[str(e[0]) + '-' + str(e[1])]==1:
                i = e[1]
                path.append(i)
                w += e[2]
    return (str(s)+'-'+str(t),path,w)

def result_info(sampleset, E, vertices):
    res = {}
    for s in sampleset.filter(lambda s: s.energy<0):
        st, path, w = path_from_sample(s,E,vertices)
        if st not in res:
            res[st]=(path,w)
    return res

def check_result(G,sampleset,E,vertices, verbose=False):
    ok = 0
    s = 0
    res = result_info(sampleset,E,vertices)
    for i in range(vertices):
        for j in range(vertices):
            if i!=j:
                if nx.has_path(G,i,j):
                    s += 1
                    p1 = [p for p in nx.all_shortest_paths(G,i,j,weight='weight')]
                    w = path_weight(G,p1[0],'weight')
                    if str(i)+'-'+str(j) in res.keys():
                        p2 = res[str(i)+'-'+str(j)]
                        if (not p2[0] in p1) and w!=p2[1]:
                            if verbose:
                                print('Path: '+str(p2[0])+' ('+str(p2[1])+'): correct: '+str(p1)+' ('+str(w)+')')
                        else:
                            ok += 1
                    else:
                        if verbose:
                            print('Path suggested: '+str(i)+'-'+str(j)+' missing: correct: '+str(p1)+' ('+str(w)+')')
    return (res,int(100*ok/s))