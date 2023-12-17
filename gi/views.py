from django.shortcuts import render

import time
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler

from qcdemo.check_result import check_result_gi
from qcdemo.qubo_functions import create_qubo_gi
from qcdemo.graphs import create_graph
from qcdemo.utils import hdata_to_json, graph_to_json

min_vertices = 5
max_vertices = 20
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

        G1, G2 = create_graph(resp['graph_type'],resp['vertices'], weight=False, directed=False, permutation=True)
        resp['gdata'] = graph_to_json(G1)
        labels = {}
        for i in range(resp['vertices']):
            for j in range(resp['vertices']):
                labels[i*resp['vertices']+j] = (i,j)
            
        Q = create_qubo_gi(G1,G2)
        bqm = dimod.BinaryQuadraticModel(Q, 'BINARY').relabel_variables(labels, inplace=False)

        result = {}
        result['edges'] = len(G1.edges)
        result['vertices'] = len(G1.nodes)
        result['qubo_size'] = Q.shape[0]
        result['logical_qubits'] = Q.shape[0]
        result['couplers'] = len(bqm.quadratic)
        result['exp_energy'] = -len(G1.edges)
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
        result['energy'] = int(sampleset.first.energy)
        result['success'] = check_result_gi(sampleset, result['exp_energy'])
        resp['result'] = result
    else:
        resp['vertices'] = 7
        resp['num_reads'] = 2000
        resp['solver'] = 'local simulator'
        resp['token'] = ''
        resp['graph_type'] = 'wheel graph'
    return render(request, 'gi/index.html', resp) 