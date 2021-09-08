import numpy as np
import dendropy
from dendropy.calculate import treecompare
import time
import sys

# to use this the RUN_PARALLEL needs to be True
# but my other changes are actually faster so do not change
# RUN_PARALLEL
#from concurrent.futures import ThreadPoolExecutor, as_completed

RUN_PARALLEL = False

def run_wRF_pair1(t1,t2):
    t2.encode_bipartitions()
    d=treecompare.weighted_robinson_foulds_distance(t1, t2, edge_weight_attr='length', is_bipartitions_updated=True)
    return d

def run_wRF_pair(a):
    t1,t2 = a
    t2.encode_bipartitions()
    d=treecompare.weighted_robinson_foulds_distance(t1, t2, edge_weight_attr='length', is_bipartitions_updated=True)
    return d

def process(tmptreelist):
    with ThreadPoolExecutor(max_workers=8) as executor:
        return  executor.map(run_wRF_pair, tmptreelist, timeout=60)


def RF_distances(n, filename_treelist):

    tic2 = time.perf_counter()
    tns = dendropy.TaxonNamespace()

    distance_matrix=np.zeros((n,n))

    f=open(filename_treelist, 'r')
    tlst = dendropy.TreeList()
    trees=tlst.get(file=f,schema="newick",taxon_namespace=tns)
    f.close()
    toc2 = time.perf_counter()
    time2 = toc2 - tic2
    print(f"\nTime of reading {n} trees= {time2}")
    #sys.exit()
    tic3 = time.perf_counter()
    for i in range(1,n):
        #    print("\ni= ", i)
        trees[i].encode_bipartitions()
        t1 = trees[i]
        if RUN_PARALLEL:
            tmptreelist = [(trees[i],trees[j]) for j in range(i)]
            dlist = process(tmptreelist)
            distance_matrix[i][:i] = list(dlist)
        else:
            for j in range(i):
                d = run_wRF_pair1(t1,trees[j])
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d
    toc3 = time.perf_counter()
    time3 = toc3 - tic3
    print(f"\nTime of generating distance matrix of {n} trees= {time3}\n\n")
#    print(distance_matrix)
    return distance_matrix
    

