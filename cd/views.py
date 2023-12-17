from django.shortcuts import render

import numpy as np
import time
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler
import dwave.inspector

from qcdemo.check_result import check_result_cd
from qcdemo.qubo_functions import create_qubo_cd
from qcdemo.graphs import create_graph
from qcdemo.utils import hdata_to_json, graph_to_json

colors=['#777777','#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc','#e5d8bd','#fddaec','#f2f2f2']

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
            return render(request, 'cd/index.html', resp) 

        if resp['communities']<min_communities or resp['communities']>max_communities:
            resp['error'] = 'communities must be '+str(min_communities)+'..'+str(max_communities)
            return render(request, 'cd/index.html', resp) 

        if resp['num_reads']>max_num_reads:
            resp['error'] = 'Maximum number fo reads is '+str(max_num_reads)
            return render(request, 'cd/index.html', resp) 

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
                return render(request, 'cd/index.html', resp) 
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
                return render(request, 'cd/index.html', resp) 
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


def result_to_colors(G, sample):
    cs = np.zeros(len(G.nodes))
    for k,v in sample.items():
        if v==1: 
            cs[k[0]]=k[1]+1
    nc = []
    for i in range(len(cs)):
        nc.append(colors[int(cs[i])])
    return nc