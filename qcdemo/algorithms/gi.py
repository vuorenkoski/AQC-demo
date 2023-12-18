from django.shortcuts import render

from dimod import BinaryQuadraticModel
import numpy as np

from qcdemo.graphs import create_graph
from qcdemo.utils import basic_stats, solve, graph_to_json, graph_to_json, colors, algorithms, graph_types

min_vertices = 5
max_vertices = 20
max_num_reads = 10000
solvers = ['local simulator', 'quantum solver', 'cloud hybrid solver']

def index(request):
    resp = {}
    resp['algorithm'] = 'Graph isomorphism'
    resp['correctness'] = 'Algorithm is tested with generated graph and the same graph having its vertices randomly permutated. '\
    'When working correctly, results should be isomorphic. Correctness is measured by how much observed '\
    'energy level differed from the correct energy level. So, for correct outcome this is 0 and more positive '\
    'this number is, more far away achieved energy level is from the correct energy level.'
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
            return render(request, 'gi/index.html', resp) 

        if resp['num_reads']>max_num_reads:
            resp['error'] = 'Maximum number fo reads is '+str(max_num_reads)
            return render(request, 'gi/index.html', resp) 

        # create graph, qubo, bqm
        G1, G2 = create_graph(resp['graph_type'],resp['vertices'], weight=False, directed=False, permutation=True)
        Q = create_qubo_gi(G1,G2)
        bqm = create_bqm_gi(Q, G1)
        result = basic_stats(G1,Q, bqm)
        result['exp_energy'] = -len(G1.edges)

        # Solve
        try:
            r, sampleset = solve(bqm,resp)
            result.update(r)
        except Exception as err:
            resp['error'] = err
            return render(request, 'algorithm.html', resp) 
        
        # Gather rest of results    
        result['energy'] = int(sampleset.first.energy)
        result['success'] = check_result_gi(sampleset, result['exp_energy'])
        resp['result'] = result

        # Create graph-data
        resp['gdata'] = {'data': graph_to_json(G1), 'colors': [colors[0] for i in range(len(G1.nodes))], 'directed':1, 'weights':1}

    else:
        # These are initial parameters for web page
        resp['vertices'] = 7
        resp['num_reads'] = 2000
        resp['solver'] = 'local simulator'
        resp['token'] = ''
        resp['graph_type'] = 'wheel graph'
    return render(request, 'algorithm.html', resp) 

def create_bqm_gi(Q, G):
    labels = {}
    vertices = len(G.nodes)
    for i in range(vertices):
        for j in range(vertices):
            labels[i*vertices+j] = (i,j)
    return BinaryQuadraticModel(Q, 'BINARY').relabel_variables(labels, inplace=False)

def create_qubo_gi(G1, G2):
    vertices = len(G1.nodes)
    E1 = [] 
    for e in G1.edges(data=True):
        E1.append((e[0],e[1]))
    E2 = [] 
    for e in G2.edges(data=True):
        E2.append((e[0],e[1]))
    p = len(E1)
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
    for i in range(vertices*vertices): 
        for j in range(i):
            Q[j,i] += Q[i,j]
            Q[i,j] = 0
    return Q

def check_result_gi(sampleset, e):
    if int(sampleset.first.energy)==e:
        return 'isomorphic'
    else:
        return 'non-isomorphic'