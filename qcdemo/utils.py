from django.http import JsonResponse
import time
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler
import dwave.inspector

algorithms = [{'name':'All pairs shortest paths', 'short':'apsp'}, 
              {'name':'Graph isomorphism', 'short':'gi'},
              {'name':'Community detection', 'short':'cd'}]

graph_types = ['path graph', 'star graph', 'cycle graph', 'complete graph', 'tree graph', 'single cycle graph', 
               'multiple cycle graph', 'bipartite graph', 'wheel graph', 'community graph', 'random graph']

colors = ['#ffffff','#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc','#e5d8bd','#fddaec','#f2f2f2']

def graph_to_json(G):
    data = []
    for e in G.edges(data=True):
        if 'weight' in e[2]:
            data.append({'source':e[0],'target':e[1],'type':e[2]['weight']})
        else:
            data.append({'source':e[0],'target':e[1]})
    return JsonResponse(data, safe=False).content.decode('utf-8')

def hdata_to_json(sampleset):
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
    datap = []
    for i in range(minv,maxv):
        if i in data.keys():
            n = data[i]
        else:
            n = 0
        datap.append({'energy':int(i),'num_occurrences':int(n)})
    return JsonResponse(datap, safe=False).content.decode('utf-8')

def basic_stats(G,Q, bqm):
    result = {}
    result['edges'] = len(G.edges)
    result['vertices'] = len(G.nodes)
    result['qubo_size'] = Q.shape[0]
    result['logical_qubits'] = Q.shape[0]
    result['couplers'] = len(bqm.quadratic)
    return result

def solve(bqm,resp):
    result = {}

    if resp['solver'] =='local simulator':
        ts = time.time()
        sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=resp['num_reads']).aggregate()
        result['time'] = int((time.time()-ts)*1000)
        resp['hdata'] = hdata_to_json(sampleset)

    elif resp['solver']=='cloud hybrid solver':
        sampleset = LeapHybridSampler(token=resp['token']).sample(bqm).aggregate()
        result['time'] = int(sampleset.info['qpu_access_time'] / 1000)

    elif resp['solver'] =='quantum solver':
        machine = DWaveSampler(token=resp['token'])
        result['chipset'] = machine.properties['chip_id']
        sampleset = EmbeddingComposite(machine).sample(bqm, num_reads=resp['num_reads']).aggregate()
        result['time'] = int(sampleset.info['timing']['qpu_access_time'] / 1000)
        result['physical_qubits'] = sum(len(x) for x in sampleset.info['embedding_context']['embedding'].values())
        result['chainb'] = sampleset.first.chain_break_fraction
        resp['hdata'] = hdata_to_json(sampleset)

    result['energy'] = int(sampleset.first.energy)
    return result, sampleset