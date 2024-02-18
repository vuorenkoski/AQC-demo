from django.shortcuts import render

from dimod import BinaryQuadraticModel
import numpy as np

from qcdemo.graphs import create_graph
from qcdemo.utils import basic_stats, solve, graph_to_json, Q_to_json, colors, algorithms, graph_types

# Parameters for UI constraints 
min_vertices = 5
max_vertices = 20
max_num_reads = 10000
solvers = ['local simulator', 'quantum solver', 'cloud hybrid solver']

def index(request):
    resp = {}
    resp['algorithm'] = 'xxxxx' # Name of the algorihtm
    resp['correctness'] = 'xxxx' # Description of how correctness is defined
    resp['algorithms'] = algorithms
    resp['solvers'] = solvers
    resp['graph_types'] = graph_types
    resp['min_vertices'] = min_vertices
    resp['max_vertices'] = max_vertices
    resp['max_num_reads'] = max_num_reads
    if request.method == "POST":
        # Get parameters
        resp['vertices'] = int(request.POST['vertices'])
        resp['num_reads'] = int(request.POST['num_reads'])
        resp['solver'] = request.POST['solver']
        resp['token'] = request.POST['token']
        resp['graph_type'] = request.POST['graph_type']

        # Check validity
        if resp['vertices']<min_vertices or resp['vertices']>max_vertices:
            resp['error'] = 'vertices must be '+str(min_vertices)+'..'+str(max_vertices)
            return render(request, 'apsp/index.html', resp) 

        if resp['num_reads']>max_num_reads:
            resp['error'] = 'Maximum number fo reads is '+str(max_num_reads)
            return render(request, 'apsp/index.html', resp) 

        # create graph, qubo, bqm
        G = create_graph(resp['graph_type'],resp['vertices'], weight=True, directed=True) # ARE WEIGHT AND DIRECTNESS NEEDED?
        Q = create_qubo(G)
        bqm = create_bqm(Q, G)
        result = basic_stats(G,Q, bqm)

        # Solve
        try:
            r, sampleset = solve(bqm,resp)
            result.update(r)
        except Exception as err:
            resp['error'] = err
            return render(request, 'algorithm.html', resp) 

        # Gather rest of results    
        resp['qdata'] = {'data': Q_to_json(Q.tolist()), 'size':len(Q)}
        result['success'] = check_result(G,sampleset)
        resp['result'] = result

        # Create graph-data
        resp['gdata'] = {'data': graph_to_json(G), 'colors': [colors[0] for i in range(len(G.nodes))], 
                         'directed':1, 'weights':1} # IF WEIGHTS AND DIRECTNESS ARE NEEDED PLACE 1
    else:
        # These are initial parameters for web page
        resp['vertices'] = 7
        resp['num_reads'] = 2000
        resp['solver'] = 'local simulator'
        resp['token'] = ''
        resp['graph_type'] = 'wheel graph'
    return render(request, 'algorithm.html', resp) 


def create_qubo(G):
    return None


def create_bqm(Q, G):
    return BinaryQuadraticModel(Q, 'BINARY') # MORE HERE IF LABELS ARE NEEDED


def check_result(G,sampleset):
    return None