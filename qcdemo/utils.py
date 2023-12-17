from django.http import JsonResponse

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