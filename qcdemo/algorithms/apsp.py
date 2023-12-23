from django.shortcuts import render

from dimod import BinaryQuadraticModel
from networkx.classes.function import path_weight
from networkx import has_path, all_shortest_paths
import numpy as np

from qcdemo.graphs import create_graph
from qcdemo.utils import basic_stats, solve, graph_to_json, Q_to_json, colors, algorithms, graph_types

min_vertices = 5
max_vertices = 20
max_num_reads = 10000
solvers = ['local simulator', 'quantum solver']

def index(request):
    resp = {}
    resp['algorithm'] = 'All pairs of shortest path'
    resp['correctness'] = 'Correctness is measured by counting what proportion of all shortest paths algorithm identified with correct sum of weights.'
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
        G = create_graph(resp['graph_type'],resp['vertices'], weight=True, directed=True)
        Q = create_qubo_apsp(G)
        bqm = create_bqm_apsp(Q, G)
        result = basic_stats(G,Q, bqm)
        resp['qdata'] = {'data': Q_to_json(Q.tolist()), 'size':len(Q)}
        
        # Solve
        try:
            r, sampleset = solve(bqm,resp)
            result.update(r)
        except Exception as err:
            resp['error'] = err
            return render(request, 'algorithm.html', resp) 

        # Gather rest of results    
        result['success'] = check_result_apsp(G,sampleset)
        result['paths'] = []
        for k,v in result_paths(G,sampleset).items():
            result['paths'].append({'nodes':k, 'path':str(v[0]), 'weight':v[1]})
        resp['result'] = result

        # Create graph-data
        resp['gdata'] = {'data': graph_to_json(G), 'colors': [colors[0] for i in range(len(G.nodes))], 'directed':1, 'weights':1}
    else:
        # These are initial parameters for web page
        resp['vertices'] = 7
        resp['num_reads'] = 2000
        resp['solver'] = 'local simulator'
        resp['token'] = ''
        resp['graph_type'] = 'wheel graph'
    return render(request, 'algorithm.html', resp) 

def create_qubo_apsp(G):
    vertices = len(G.nodes)
    E = [] 
    for e in G.edges(data=True):
        E.append((e[0],e[1],e[2]['weight']))

    p = 1
    for e in E:
        p += e[2]

    edges = len(E)
    Q = np.zeros((2*vertices + edges, 2*vertices + edges))

    # Constraints 1 and 2
    for i in range(vertices):
        for j in range(i+1, vertices):
            Q[i,j] += p
            Q[vertices+i,vertices+j] += p
        
    # Constraint 3
    for i in range(vertices):
        Q[i,i+vertices] += p

    # Constraint 4
    for i in range(edges):
        Q[E[i][0],vertices*2+i] -= p
        Q[E[i][1],vertices*2+i] += p
 
    # Constraint 5
    for i in range(edges):
        Q[vertices+E[i][1],vertices*2+i] -= p
        Q[vertices+E[i][0],vertices*2+i] += p

    # Constraint 6
    for i in range(edges):
        for j in range(i+1,edges):
            if E[i][0]==E[j][0] or E[i][1]==E[j][1]:
                Q[vertices*2+i,vertices*2+j] += p

    # Constraint 7
    for i in range(edges):
        Q[vertices*2+i,vertices*2+i] +=p
        for j in range(i+1,edges):
            if E[i][1]==E[j][0] or E[i][0]==E[j][1]:
                Q[vertices*2+i,vertices*2+j] -= p

    # Constraint 8 
    for i in range(edges):
        Q[vertices*2+i,vertices*2+i] += E[i][2]

    return Q

def create_bqm_apsp(Q, G):
    labels = {}
    vertices = len(G.nodes)
    for i in range(vertices):
        labels[i]='s'+str(i)
        labels[vertices+i]='t'+str(i)   
    for i,e in enumerate(G.edges):
        labels[vertices*2+i] = str(e[0]) + '-' + str(e[1])
    return BinaryQuadraticModel(Q, 'BINARY').relabel_variables(labels, inplace=False)

def xy_from_label(e):
    x = 0
    y = 0
    i = len(e)-1
    d = 1
    while e[i]!='-':
        y += d*int(e[i])
        d *= 10 
        i -= 1
    d = 1
    i -= 1
    while i>=0:
        x += d*int(e[i])
        d *= 10
        i -= 1 
    return (x,y)

def path_from_sample(sample,G):
    vertices = len(G.nodes)
    s = 0
    si = 0
    t = 0
    ti = 0
    w = 0
    for v in range(vertices):
        if sample['s'+str(v)]==1:
            s = v
            si += 1
        if sample['t'+str(v)]==1:
            t = v
            ti += 1
    if s==t or si!=1 or ti!=1: # only one s and t allowed, not the same
        return (None,None,None)
    i = s
    path = [i]
    vv = 1
    while i!=t:
        xx = i
        for p in sample:
            if p[0]!='s' and p[0]!='t' and sample[p]==1:
                x,y = xy_from_label(p)
                if x==i:
                    path.append(y)
                    w += G[x][y]['weight']
                    i = y
                    break
        vv += 1
        if vv>vertices:
            return (None,None,None)
    return (str(s)+'-'+str(t),path,w)

def result_paths(G, sampleset): # return all legal shrotest paths from sampleset
    res = {}
    for s in sampleset.filter(lambda s: s.energy<0):
        st, path, w = path_from_sample(s,G)
        if st!=None:
            if st not in res:
                res[st]=(path,w)
            else:
                if res[st][1]>w:
                    res[st]=(path,w)
    return res

def check_result_apsp(G,sampleset):
    ok = 0
    s = 0
    res = result_paths(G, sampleset)
    for i in G.nodes:
        for j in G.nodes:
            if i!=j and has_path(G,i,j):
                s += 1
                sp = [p for p in all_shortest_paths(G,i,j,weight='weight')] # correct shorstest paths for this pair
                w1 = path_weight(G,sp[0],'weight') # correct weight
                if str(i)+'-'+str(j) in res.keys():  # Does result have path between s-t
                    path,w2 = res[str(i)+'-'+str(j)]
                    if (path in sp) and w1==w2:      # Is path among correct paths and are weights same
                        ok += 1
    return str(int(100*ok/s))+'%'