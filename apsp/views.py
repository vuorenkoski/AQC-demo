from django.shortcuts import render

import time
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler

from qcdemo.check_result import check_result_apsp, result_paths
from qcdemo.qubo_functions import create_qubo_apsp
from qcdemo.graphs import create_graph
from qcdemo.utils import hdata_to_json, graph_to_json

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
            resp['hdata'] = hdata_to_json(sampleset)
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
    return render(request, 'apsp/index.html', resp) 
