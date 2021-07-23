#!/usr/bin/env python
#
# (c) Tara Khodaie, Tallahassee FL 2021
#
DEBUG=False
import sys
import numpy as np

from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent
sys.path.append(str(file.parent))
#try:
#    sys.path.remove(str(parent))
#except ValueError: # Already removed
#    pass
#print(sys.path)
    
import splittree
import subtree
import tree
import itertools
import copy

DEBUG=True

def pathtrees(treefile, terminallist, numpathtrees):
    '''
    Generates a path between two trees using the geodesic
    '''
    precision = 4    
    if DEBUG:
        print(f'\n\nReading Data +++++++++++++++++++++++++++++++++++++++\n')    
    #NOTE:  Results_subtrees has all subtrees of T1 and T2.
    #       Each subtree has all information of it including
    #       tips,edges,tipslength, edgelengths,
    #       and supports(supports if two corresponding subtrees are disjoint)
    Results_subtrees, tip_dict , edge_dict = subtree.subtrees(treefile,terminallist)
    T1 = Results_subtrees[0]
    if DEBUG:
        print("T1",T1)
    T2 = Results_subtrees[1]
    disjoint_indices = Results_subtrees[2]
    if DEBUG:
        print("\nTree1 subtrees :\n",T1)
        print("\n\nTree2 subtrees :\n",T2)
        print("\n\ndisjoint_indices :\n",disjoint_indices)
    
    T1_tip_dict = tip_dict[0]
    T1_edge_dict = edge_dict[0]
    T2_tip_dict = tip_dict[1]
    T2_edge_dict = edge_dict[1]
    if DEBUG:
        print("\n\nTree1 tips dictionary :\n",T1_tip_dict)
        print("\n\nTree1 edges dictionary :\n",T1_edge_dict)
        print("\n\nTree2 tips dictionary :\n",T2_tip_dict)
        print("\n\nTree2 edges dictionary :\n",T2_edge_dict)
    
    File = open(treefile, "r")
    file = File.readlines()
    treelist=np.array([s.replace('\n', '') for s in file])
    
    if DEBUG:
        print(f'\nCreate PathTrees ++++++++++++++++++\n')
    
    Lamda = np.linspace(0.0, 1.0, num=numpathtrees)
    Lamda = [round(elem,precision) for elem in Lamda ]        # 4 precision
    #print("*****TEST******", Lamda)
    #myfile = open(mypathtrees, 'w')
    thetreelist = []
    #for l, lamda in enumerate(Lamda):     ****NEW****
    for l, lamda in enumerate(Lamda[1:-1]):
        if DEBUG:
            print(f'\nPathTree #{l}, lamda={lamda} ====================\n')
        T1_path = copy.deepcopy(T1)
        for num in range(len(T1_path)):
            if DEBUG:
                print(f"\n\nnum = {num}")
            if num in (disjoint_indices[0]):
                if DEBUG:
                    print(f'\n~~~~~~~~~~~~~~~~~~~~~~ subtree #{num}    ,    lamda{lamda} ~~~~~~~~~~~~~~~~~~~~~~~~~\n')
                    print(f"\n########  NOTE:  num #{num} IS inside disjoint_indices ########\n\n ")
    #            print(f"\nFor subtree #{num}, for lambda {lamda}:\n\n ")
                lambda_limits, epsilon = subtree.path_legs(num,  T1, T2, T1_edge_dict, T2_edge_dict)
    #            print(f"\n===> lambda_limits = {lambda_limits}")
    #            print(f"\n===> epsilon :\n {epsilon}")
                edited_subtree = subtree.pathtree_edges(num, T1, T2, T1_edge_dict, T2_edge_dict, lamda, lambda_limits, epsilon)
                #            print(f"\n===> editted path subtree : \n {edited_subtree}\n")        #------*************------
                T1_path[num] = edited_subtree
    #            print("Origional T1[num] :\n", T1[num])    #test
    #            print("\nT1_path[num] :\n", T1_path[num])      #test
            else:
                if DEBUG:
                    print(f'\n~~~~~~~~~~~~~~~~~~~ common subtree #{num}    ,    lamda{lamda} ~~~~~~~~~~~~~~~~~~~~~\n')
                    print(f"\n########  NOTE:  num #{num} IS NOT inside disjoint_indices ########\n\n ")     #???????Here not work
    #            print("++++++TEST++++++++++", T1[num][-2], T2[num][-2])
                T1_path[num][-2] = list( (1-lamda)*(np.array(T1[num][-2]))  + lamda*(np.array(T2[num][-2]) ) )   # length of tips(leaves) " (1-lambda)*e_T +lambda*e_T' "
                T1_path[num][-2] = [round(elem,precision) for elem in T1_path[num][-2] ]
                T1_path[num][-1] = list( (1-lamda)*(np.array(T1[num][-1]))  + lamda*(np.array(T2[num][-1]) ) )   #length of each common edge " (1-lambda)*e_T +lambda*e_T' "
                T1_path[num][-1] = [round(elem,precision) for elem in T1_path[num][-1] ]
                if DEBUG:
                    print("starting common subtree :\n", T1[num])    #test
                    print("endinging common subtree :\n", T2[num])    #test
                    print("\nEditted starting common subtree :\n", T1_path[num])      #test
        if DEBUG:
            print(f'\n~~~~~~~~~~~~~~~~~~~~~~ generated pathtree #{l} ~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            print("Origional T1 :\n", T1)
            print("\nGenerated pathtree from T1 :\n", T1_path)
            #test
            print("\n\n\nnewick of T1 :\n", treelist[0])
            print("\nnewick of T2 :\n", treelist[1])     #TARA
        sub_newicks , newick = subtree.Sub_Newicks(T1_path, disjoint_indices[0] )
        if DEBUG:
            print("\nsubtree newicks of path tree :\n",sub_newicks)
            print("\nnewick of pathtree :\n", newick)
        #myfile.write(newick + '\n')
        thetreelist.append(newick)
    #myfile.close()
    return thetreelist
    

if __name__ == "__main__":

    if len(sys.argv)<2:
        print("pathtrees.py treefile terminalist")
    treefile = sys.argv[1]
    terminallist = sys.argv[2]
    numpathtrees=10
    mypathtrees='output_path'
    mypathtrees = pathtrees(treefile, terminallist, numpathtrees)
    print("Standalone test:\nlook at",mypathtrees)
