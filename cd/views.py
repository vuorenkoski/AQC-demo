from django.shortcuts import render

import numpy as np
import time
import random
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler
import networkx as nx

from qcdemo.check_result import check_result_cd
from qcdemo.qubo_functions import create_qubo_cd
from qcdemo.graphs import create_graph
from qcdemo.utils import hdata_to_json, graph_to_json

colors=["#fbb4ae","#b3cde3","#ccebc5","#decbe4","#fed9a6","#ffffcc","#e5d8bd","#fddaec","#f2f2f2"]

min_vertices = 5
max_vertices = 20
min_communities = 1
max_communities = 10
max_num_reads = 10000
solvers = ['local simulator', 'quantum solver', 'cloud hybrid solver']
graph_types = ['path graph', 'star graph', 'cycle graph', 'complete graph', 'tree graph', 'single cycle graph', 
               'multiple cycle graph', 'bipartite graph', 'wheel graph', 'community graph', 'random graph']


def index(request):
    resp = {}
    resp['solvers'] = solvers
    resp['graph_types'] = graph_types
    resp['min_vertices'] = min_vertices
    resp['max_vertices'] = max_vertices
    resp['min_communities'] = min_communities
    resp['max_communities'] = max_communities
    resp['max_num_reads'] = max_num_reads
    if request.method == "POST":
        resp['vertices'] = int(request.POST['vertices'])
        resp['num_reads'] = int(request.POST['num_reads'])
        resp['solver'] = request.POST['solver']
        resp['token'] = request.POST['token']
        resp['graph_type'] = request.POST['graph_type']
        resp['communities'] = int(request.POST['communities'])

        if resp['vertices']<min_vertices or resp['vertices']>max_vertices:
            resp['error'] = 'vertices must be '+str(min_vertices)+'..'+str(max_vertices)
            return render(request, 'apsp/index.html', resp) 

        if resp['communities']<min_communities or resp['communities']>max_communities:
            resp['error'] = 'communities must be '+str(min_communities)+'..'+str(max_communities)
            return render(request, 'apsp/index.html', resp) 

        if resp['num_reads']>max_num_reads:
            resp['error'] = 'Maximum number fo reads is '+str(max_num_reads)
            return render(request, 'apsp/index.html', resp) 

        G = create_graph(resp['graph_type'],resp['vertices'], weight=True, directed=False)
        labels = {}
        for i in range(len(G.nodes)):
            for j in range(resp['communities']):
                labels[i*resp['communities'] + j]=(i,j)

        Q = create_qubo_cd(G, resp['communities'])
        bqm = dimod.BinaryQuadraticModel(Q, 'BINARY').relabel_variables(labels, inplace=False)

        result = {}
        result['edges'] = len(G.edges)
        result['vertices'] = len(G.nodes)
        result['qubo_size'] = Q.shape[0]
        result['logical_qubits'] = Q.shape[0]
        result['couplers'] = len(bqm.quadratic)
        if resp['solver'] =='local simulator':
            ts = time.time()
            sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=resp['num_reads']).aggregate()
            result['time'] = int((time.time()-ts)*1000)
            resp['hdata'] = hdata_to_json(sampleset)
        elif resp['solver']=='cloud hybrid solver':
            try:
                sampleset = LeapHybridSampler(token=resp['token']).sample(bqm).aggregate()
                result['time'] = int(sampleset.info['qpu_access_time'] / 1000)
            except Exception as err:
                resp['error'] = err
                return render(request, 'apsp/index.html', resp) 
        elif resp['solver'] =='quantum solver':
            try:
                machine = DWaveSampler(token=resp['token'])
                result['chipset'] = machine.properties['chip_id']
                sampleset = EmbeddingComposite(machine).sample(bqm, num_reads=resp['num_reads']).aggregate()
                result['time'] = int(sampleset.info['timing']['qpu_access_time'] / 1000)
                result['physical_qubits'] = sum(len(x) for x in sampleset.info['embedding_context']['embedding'].values())
                result['chainb'] = sampleset.first.chain_break_fraction
                resp['hdata'] = hdata_to_json(sampleset)
            except Exception as err:
                resp['error'] = err
                return render(request, 'apsp/index.html', resp) 
        resp['gdata'] = graph_to_json(G)
        resp['gcolors'] = result_to_colors(G, sampleset.first.sample)
        result['energy'] = int(sampleset.first.energy)
        result['success'] = check_result_cd(G,sampleset,resp['communities'])
        resp['result'] = result
    else:
        resp['vertices'] = 7
        resp['num_reads'] = 2000
        resp['solver'] = 'local simulator'
        resp['token'] = ''
        resp['graph_type'] = 'community graph'
        resp['communities'] = 4
    return render(request, 'cd/index.html', resp) 

def index2(request):
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

def result_to_colors(G, sample):
    cs = np.zeros(len(G.nodes))
    for k,v in sample.items():
        if v==1: 
            cs[k[0]]=k[1]+1
    nc = []
    for i in range(len(cs)):
        nc.append(colors[int(cs[i])])
    return nc