from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

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

from qcdemo.check_result import check_result_apsp, result_paths
from qcdemo.qubo_functions import create_qubo_apsp
from qcdemo.graphs import create_graph

min_vertices = 5
max_vertices = 20
max_num_reads = 10000
solvers = ['local simulator', 'quantum solver']
graph_types = ['path graph', 'star graph', 'cycle graph', 'complete graph', 'tree graph', 'single cycle graph', 
               'multiple cycle graph', 'bipartite graph', 'wheel graph', 'community graph', 'random graph']

def index(request):
    resp = {}
    resp['solvers'] = solvers
    resp['graph_types'] = graph_types
    resp['min_vertices'] = min_vertices
    resp['max_vertices'] = max_vertices
    resp['max_num_reads'] = max_num_reads
    resp['data'] = JsonResponse([], safe=False).content.decode('utf-8')
    if request.method == "POST":
        resp['vertices'] = int(request.POST['vertices'])
        resp['num_reads'] = int(request.POST['num_reads'])
        resp['solver'] = request.POST['solver']
        resp['token'] = request.POST['token']
        resp['graph_type'] = request.POST['graph_type']

        if resp['vertices']<min_vertices or resp['vertices']>max_vertices:
            resp['error'] = 'vertices must be '+str(min_vertices)+'..'+str(max_vertices)
            return render(request, 'apsp/index.html', resp) 

        if resp['num_reads']>max_num_reads:
            resp['error'] = 'Maximum number fo reads is '+str(max_num_reads)
            return render(request, 'apsp/index.html', resp) 

        G = create_graph(resp['graph_type'],resp['vertices'], weight=True, directed=True)
        resp['gdata'] = graph_to_json(G)
        labels = {}
        for i in range(resp['vertices']):
            labels[i]='s'+str(i)
            labels[resp['vertices']+i]='t'+str(i)   
        for i,e in enumerate(G.edges):
            labels[resp['vertices']*2+i] = str(e[0]) + '-' + str(e[1])
            
        
        Q = create_qubo_apsp(G)
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
            resp['histogram'] = print_histogram(sampleset, fig_size=5)
        elif resp['solver'] =='cloud hybrid solver':
            try:
                sampleset = LeapHybridSampler(token=resp['token']).sample(bqm).aggregate()
                result['time'] = int(sampleset.info['qpu_access_time'] / 1000)
                resp['histogram'] = None
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
                resp['histogram'] = print_histogram(sampleset, fig_size=5)
            except Exception as err:
                resp['error'] = err
                return render(request, 'apsp/index.html', resp) 
        result['energy'] = int(sampleset.first.energy)
        result['success'] = check_result_apsp(G,sampleset)
        result['paths'] = []
        for k,v in result_paths(G,sampleset).items():
            result['paths'].append({'nodes':k, 'path':str(v[0]), 'weight':v[1]})
        resp['result'] = result
    else:
        resp['vertices'] = 7
        resp['num_reads'] = 2000
        resp['solver'] = 'local simulator'
        resp['token'] = ''
        resp['graph_type'] = 'wheel graph'

        resp['graph'] = None
        resp['histogram'] = None
        resp['result'] = {}
    return render(request, 'apsp/index.html', resp) 

def graph_to_json(G):
    data = []
    for e in G.edges(data=True):
        data.append({'source':e[0],'target':e[1],'type':e[2]['weight']})
#    JsonResponse([{"source":0,"target":1,"type":10}, {"source":0,"target":2,"type":5}], safe=False)
    return JsonResponse(data, safe=False).content.decode('utf-8')



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

