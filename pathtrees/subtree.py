#!usr/bin/env python
DEBUG=False
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent
sys.path.append(str(file.parent))
#print(sys.path)

import random
import math
import numpy as np
import tree
import itertools
import splittree
import copy


import bifurcating       

precision = 4

def subtrees(treelist, terminallist):
    File = open(treelist, "r")
    file = File.readlines()
    T=np.array([s.replace('\n', '') for s in file])

    Edges_list=[]
    EdgeLengths_list=[]

    Tip_list=[]
    TipLengths_list=[]

    for k in [0,1]:
#        print(f'\n================================= Tree {k+1} ===================================\n')
        p = tree.Node()
        p.name = 'root'
        mytree = tree.Tree()
        mytree.root=p
        mytree.myread(T[k],p)
        
        edges=[]
        edgelengths=[]
        Edges, Edgelengths =  mytree.get_edges(p, edges,edgelengths)
        Edges = [ sorted(edge) for edge in Edges ]

        tips=[]
        tipslength=[]
        tips, tipslength = mytree.getTips(p, tips, tipslength)
        tipnames=[tips[i].name for i in range(len(tips))]

        Edges.append(sorted(tipnames))    #add root as an edge with length zero
        Edgelengths.append(0.0)
#        print(f"Tree{k+1} :", T[k])
#        print("\nEdges :", Edges)
#        print("\nEdges length :", Edgelengths)
#        print("\nTips :", tipnames)
#        print("\nTips length :", tipslength)
        Edges_list. append(Edges)
        EdgeLengths_list. append(Edgelengths)

        Tip_list. append(tipnames)
        TipLengths_list. append(tipslength)


#    print(f'\n=================== Commom & Uncommon edges with lengths ====================\n')

    Common_Edges = [edge for edge in Edges_list[0] if edge in Edges_list[1]]

    Common_indices = []
    Common_Edges_length = []
    Uncommon_Edges = []
    Uncommon_indices = []
    Uncommon_Edges_length = []
    for i in range(2):
        Common_indices.append([Edges_list[i].index(edge) for edge in Common_Edges])
        Common_Edges_length.append([EdgeLengths_list[i][l] for l in Common_indices[i]] )

        Uncommon_Edges.append([x for x in Edges_list[i] if x not in Common_Edges])
        Uncommon_indices.append([Edges_list[i].index(edge) for edge in Uncommon_Edges[i]])
        Uncommon_Edges_length.append([EdgeLengths_list[i][l] for l in Uncommon_indices[i]] )
#    print("\nCommon edges :", Common_Edges)
#    print("Common edges length :", Common_Edges_length)
#    print("\n\nUncommon edges of Tree1:", Uncommon_Edges[0])
#    print("Uncommon edges length of Tree1:", Uncommon_Edges_length[0])
#    print("\n\nUncommon edges of Tree2:", Uncommon_Edges[1])
#    print("Uncommon edges length of Tree2:", Uncommon_Edges_length[1])
    sub_list=[]
    Results=[]
    sub_dict=[]
    edge_dict=[]
    tip_dict=[]
    for t in range(2):    #two trees
#        print(f'\n*******************************  Tree{t+1}   ******************************************\n')

        edge_list = sorted(Edges_list[t], key=len).copy()    #all edges of tree
        common_list = sorted(Common_Edges, key=len)      #common edges of tree
        uncommon_list = ([sorted(uncommonedges, key=len) for uncommonedges in Uncommon_Edges])[t]      #uncommon edges of tree

        dict_edges = dict(zip([str(i) for i in Edges_list[t]] , EdgeLengths_list[t]))
        dict_tips = dict(zip([str(i) for i in Tip_list[t]]  , TipLengths_list[t]))
        dict_subs ={}       #*****NEW*******
#        print("\nDictionary of edges :\n",dict_edges.items())
#        print("\n\nDictionary of tips :\n",dict_tips.items())

        edges_new = edge_list.copy()
        common_new = common_list.copy()
        uncommon_new = uncommon_list.copy()
#        print("\ncommon list :", common_list)
#        print("\nuncommon list :", uncommon_list)


        subT_edges = []      #disjoint subtrees
        subT_tips = []
        subT_edges_length = []
        subT_tips_length = []
        i =0
        while (len(common_new)>0):
            edge = common_new[0]
#            print("\n","~"*20, "\n")
#            print("\nedge :", edge)

            if len(edge)==2:
                subtree = [edge]
                dict_tips.update( {'subT' + str(i) : dict_edges[str(edge)]} )
            elif len(edge)>2:
                subtree = [edge]
                dict_tips.update( {'subT' + str(i) : dict_edges[str(edge)]} )

                for j, item in enumerate(common_new):     #for some reason this loop does not work!!!!!!!!!!
                    if set(item) < set(edge):
                        subtree.append('subT'+ str(j))

                unwanted=[]
                for j, item in enumerate(uncommon_new):
                    if set(item) <= set(edge):
                        subtree.append(item)
                        unwanted.append(item)
                uncommon_new = [ele for ele in uncommon_new if ele not in unwanted]


            for j, item in enumerate(common_new):
                if set(edge) < set(item):
                    common_new[j]= ['subT'+ str(i)]
                    common_new[j].extend(list(set(item)-set(edge)))
                    dict_edges.update( {str(common_new[j]) : dict_edges[str(item)]} )

            for j, item in enumerate(uncommon_new):
                if set(edge) < set(item):
                    uncommon_new[j]= ['subT'+ str(i)]
                    uncommon_new[j].extend(list(set(item)-set(edge)))
                    dict_edges.update( {str(uncommon_new[j]) : dict_edges[str(item)]} )

            common_new.remove(edge)
            common_new = sorted(common_new, key=len)

            edges_new = sorted(common_new + uncommon_new, key=len)

            subT_edges.append(subtree)
            subT_tips.append(edge)

            i=i+1
#            print("\n\n\nsubtree =", subtree)
#            print("\nsubtree tips =", edge)
#            print("\ncommon_new :", common_new)
#            print("\nuncommon_new :", uncommon_new)
#            print("\nedges_new :", edges_new)

#        print(f"\n******* dict_subs{t+1} ******** :\n", dict_subs)      #*****NEW*******

#        print(f'\n\n*****************************  Tree{t+1} subtrees  *******************************\n')
#        print("\nsubtree's edges list :\n", subT_edges)
#        print("\n\nsubtree's tips list :\n", subT_tips)
#        print("\n\nDictionary of edges :\n",dict_edges.items())
#        print("\n\nDictionary of tips :\n",dict_tips.items())

        subT_edges_length=[]
        subT_tips_length =[]
        for i in range(len(subT_tips)):
            subT_tips_length.append([dict_tips[item] for item in subT_tips[i]])
            subT_edges_length.append([dict_edges[str(item)] for item in subT_edges[i]])
#        print("\nsubtrees tip list :\n", subT_tips)
#        print("\nsubtrees tip length list :\n",subT_tips_length)
#        print("\nsubtrees edge list :\n",subT_edges)
#        print("\nsubtrees edge length list :\n",subT_edges_length)

        result_tree = [str(subT_tips), str(subT_edges), str(subT_tips_length), str(subT_edges_length)]

        s = [subT_tips, subT_edges, subT_tips_length, subT_edges_length]
        Results.append(s)
        
        sub_list.append(subT_edges)
        
        sub_dictionary = {}
        for i in range(len(subT_tips)):
            sub_dictionary['subT' + str(i)] = subT_tips[i]
#        print(f"\nsub_dictionary Tree{t+1} :\n", sub_dictionary)
        sub_dict.append(sub_dictionary)

        edge_dict.append(dict_edges)
        tip_dict.append(dict_tips)

#    print(f'\n\n******************************************************************************\n')

    sub_list = [[sorted(sub_list[i][j], key=len)[::-1] for j in range(len(sub_list[i]))] for i in range(2) ]

    for s, subT in enumerate([sub_list[0], sub_list[1]]):
#        print(f"Origional Tree{s+1} subtrees : \n", subT)
        for i in range(1,len(subT)):
            for j, item in enumerate(subT[i]):
                for k in range(len(subT)):
                    item = [subT[k][0] if x==str('subT' + str(k)) else [x] for x in item]
                    item = list(itertools.chain.from_iterable(item))
                    subT[i][j]=item
#        print(f"\nEditted Tree{s+1} subtrees : \n", subT,"\n\n")

#    print(f'\n*************************  Extract GTP Informations ***************************\n')

    Combinatorial=[]
    start_tree=[]
    end_tree=[]

    mylines = []
    with open (terminallist, 'rt') as myfile:
        for myline in myfile:
            mylines.append(myline.rstrip('\n'))

    index=[]
    for i,line in enumerate(mylines):
        if line.lower().find("Starting tree edges:".lower()) != -1:       # If a match is found
            index.append(i+2)
        if line.lower().find("Target tree edges:".lower()) != -1:
            index.append(i+2)
        if line.lower().find("Leaf contribution squared".lower()) != -1:
            index.append(i)
#    print( "indexes :", index)

    for line in mylines[index[0]:index[1]-3]:
        start_tree.append(line)
    start_tree=[s.split('\t\t') for s in start_tree]
    start_tree=[[s[0] , s[-1].split(',')] for s in start_tree]

    for line in mylines[index[1]:index[2]-1]:
        end_tree.append(line)
    end_tree=[s.split('\t\t') for s in end_tree]
    end_tree=[[s[0] , s[-1].split(',')] for s in end_tree]

    for line in mylines[index[2]:]:
        if line.lower().find("Combinatorial type:".lower()) != -1:
            Combinatorial.append(line)

    Combinatorial=[(support.split(': '))[1].split(';')[:-1] for support in  Combinatorial]

    supports=[]
    for s in Combinatorial:
        A=[];B=[]
        for item in s:
            item = item.rstrip().split('/')
            A.append(item[0][1:-1].split(','))
            B.append(item[1][1:-1].split(','))
        supports.append([A,B])
#    print( "Starting Tree :\n", start_tree)
#    print( "\n\nEnding Tree :\n", end_tree)
#    print( "\n\nSupports list :\n", supports)

    dict_GTP1 = dict(zip([str(i[0]) for i in start_tree] , [str(i[1]) for i in start_tree]))
    dict_GTP2 = dict(zip([str(i[0]) for i in end_tree] , [str(i[1]) for i in end_tree]))
#    print("\n\nDictionary of starting tree :\n",dict_GTP1.items())
#    print("\n\nDictionary of ending tree :\n",dict_GTP2.items())

#    print(f'\n\n******************************************************************************\n')

    A_list = []
    B_list = []

    for support in supports:
        A = support[0]
        B = support[1]
        A = [[dict_GTP1[a[i]] for i in range(len(a))] for a in A]
        B = [[dict_GTP2[b[i]] for i in range(len(b))] for b in B]
        item_A = [[eval(s.rstrip()) for s in l] for l in A]
        item_B = [[eval(s.rstrip()) for s in l] for l in B]

        A_list.append(item_A)
        B_list.append(item_B)
#    print( "\nA_list :\n", A_list)
#    print( "\n\nB_list :\n", B_list)

    support_list = [A_list , B_list]


#    print(f'\n\n******************************************************************************\n')

    flatten = itertools.chain.from_iterable

    flatten_list = []
    for i in range(2):
        flat=[]
        for item in support_list[i]:
#            print( "\n\nitem***** :\n", item)
            result = list(flatten(list(flatten(item))))
            result = list(set(result))      #to remove duplicates in the lists
#            print( "\n\nresult***** :\n", result)
            flat.append(result)
        flatten_list.append(flat)
#    print( "\n\nflatten_list :\n", flatten_list)
    flatten_A = flatten_list[0]
    flatten_B = flatten_list[1]
#    print( "\nflatten_A :\n", flatten_A)
#    print( "\n\nflatten_B :\n", flatten_B)


#    print(f'\n\n***************  Add GTP Supports to "Results/subtrees file"  ****************\n')
#
#    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#    print(f'Tree1 information : Results[0]\n')
#    print(f'Tree2 information : Results[1]\n')
#    print(f'Each result has information of all tree subtrees in order of : subT_tips , subT_edges , subT_tips_length , subT_edges_lentgh')
#    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    T1 = Results[0]
    T1 = [[item[i] for item in T1] for i in range(len(T1[0]))]
    T2 = Results[1]
    T2 = [[item[i] for item in T2] for i in range(len(T2[0]))]
#    print("\nTree1 subtrees :\n",T1)
#    print("\n\nTree2 subtrees :\n",T2)
#    print( "\nResults :\n", Results)

    T1_editted = sub_list[0]
    T2_editted = sub_list[1]
#    print("\n\nEditted Tree1 subtrees (edges) :\n", T1_editted)
#    print("\n\nEditted Tree2 subtrees (edges) :\n", T2_editted)


    disjoint_indices_T1=[]
    for i, item in enumerate(flatten_A):
        for j, support in enumerate(T1_editted):
            if len(support)>1:
                if any (set(x) == set(item) for x in support):
                    disjoint_indices_T1.append(j)
                    T1[j].append(A_list[i])
#    print("\nTree1 subtrees with supports:\n",T1)
#    print("\ndisjoint_indices_T1 : ",disjoint_indices_T1)


    disjoint_indices_T2=[]
    for i, item in enumerate(flatten_B):
        for j, support in enumerate(T2_editted):
            if len(support)>1:
                if any (set(x) == set(item) for x in support):
                    disjoint_indices_T2.append(j)
                    T2[j].append(B_list[i])
#    print("\nTree2 subtrees with supports:\n",T2)
#    print("\ndisjoint_indices_T2 : ",disjoint_indices_T2)


#    print(f'\n\n***************  Test : GTP Support list based on subs"  ****************\n')

    Tree1 = Results[0]
    Tree1 = [[item[i] for item in Tree1] for i in range(len(Tree1[0]))]
    Tree2 = Results[1]
    Tree2 = [[item[i] for item in Tree2] for i in range(len(Tree2[0]))]
#    print("\nTree1 subtrees :\n",Tree1)
#    print("\n\nTree2 subtrees :\n",Tree2)
#    print("\n\n\nA_list :\n",A_list)

    A_list_editted = A_list.copy()
#    M1 = list(sorted(sub_dict[0].keys()))     #EDIT
    M1 = list(sub_dict[0].keys())
#    print("M1 :", M1)
    for c in A_list_editted:
        for b in c:
            for m in M1:
                for i, a in enumerate(b):
                    if set(sub_dict[0][m]) <= set(a):
                        b[i]=[m]
                        b[i].extend(list(set(a)-set(sub_dict[0][m])))
#    print("\nA_list_editted :\n",A_list_editted)
#    print("\n\n\nB_list :\n",B_list)
    B_list_editted = B_list.copy()
#    M2 = list(sorted(sub_dict[1].keys()))     #EDIT
    M2 = list(sub_dict[1].keys())
#    print("M2 :", M2)
    for c in B_list_editted:
        for b in c:
            for m in M2:
                for i, a in enumerate(b):
                    if set(sub_dict[0][m]) <= set(a):
                        b[i]=[m]
                        b[i].extend(list(set(a)-set(sub_dict[1][m])))
#    print("\nB_list_editted :\n",B_list_editted)

    Tree1_editted = sub_list[0]
    Tree2_editted = sub_list[1]
#    print("\n\nEditted Tree1 subtrees (edges) :\n", Tree1_editted)
#    print("\n\nEditted Tree2 subtrees (edges) :\n", Tree2_editted)
    
    disjoint_indices_Tree1=[]
    for i, item in enumerate(flatten_A):
        for j, support in enumerate(Tree1_editted):
            if len(support)>1:
                if any (set(x) == set(item) for x in support):
                    disjoint_indices_Tree1.append(j)
                    Tree1[j].append(A_list_editted[i])
#    print("\n\n\nTree1 subtrees with supports:\n",Tree1)
#    print("\ndisjoint_indices_Tree1 : ",disjoint_indices_Tree1)

    disjoint_indices_Tree2=[]
    for i, item in enumerate(flatten_B):
        for j, support in enumerate(Tree2_editted):
            if len(support)>1:
                if any (set(x) == set(item) for x in support):
                    disjoint_indices_Tree2.append(j)
                    Tree2[j].append(B_list_editted[i])
#    print("\n\n\nTree2 subtrees with supports:\n",Tree2)
#    print("\ndisjoint_indices_Tree2 : ",disjoint_indices_Tree2)

    Results_subtrees =[Tree1, Tree2, [sorted(disjoint_indices_Tree1) , sorted(disjoint_indices_Tree2)]]
#    np.savetxt ('Results_subtrees.txt', Results_subtrees,  fmt='%s' , delimiter=', ')

    return(Results_subtrees, tip_dict, edge_dict)





def Sub_Newicks(T, disjoint_indices):       # ***** DEBUG *****
    print("***TEST : len(T) ****", len(T))
    sub_newicks_list  = []
    for i in range(len(T)): #****************  BIFURCATING  **************
        #        print(f"\n=========== TEST 1 ========\n {T[i]}\n")
        T1 = bifurcating.bifurcating(T[i][0], T[i][1][1:], T[i][2], T[i][3][1:])
        #        print(f"\n=========== TEST 2 ========\n T1:\n{T1}\n\n\n")
        if i in disjoint_indices:
            newick_sub = splittree.print_newick_string(T1[0], T1[1], T1[2], T1[3] )
            print(f"subT{i} --> disjoint :", newick_sub)
        if i not in disjoint_indices:
            newick_sub  = '('+str(T1[0][0])+':'+str(T1[2][0])+','+ str(T1[0][1])+':'+str(T1[2][1])+')'+':0.0'
            print(f"subT{i} --> NOT disjoint :", newick_sub)
        sub_newicks_list.append(newick_sub)
    if DEBUG:
        print("sub_newicks list :", sub_newicks_list)
    newick = sub_newicks_list[-1]
    for i in range(len(sub_newicks_list)-2, -1,-1):
        newick = newick.replace(str('subT' + str(i)), sub_newicks_list[i][:-4])
    #    print("\n\nnewick : ", newick)
    newick = newick+';'
    #    np.savetxt ('newick', newick,  fmt='%s')
    return(sub_newicks_list, newick)





# subT1, subT2 are two disjoint trees ( we know all information of them including tips,edges,tipslength, edgelengths, supports)
def path_legs(num, T1, T2, T1_edge_dict, T2_edge_dict):      # lamda a number in [0,1]
    subT1 = T1[num]    # subT1, subT2 should be disjoint trees  (num: the numeber of disjoint pair trees that we want to study)
    subT2 = T2[num]
    A = subT1[-1]
    B = subT2[-1]
    print(f"\n===> starting subtree :\n {subT1}")
    print(f"\n===> ending subtree :\n {subT2}")
#    print("\n\n\nsubT1 = ", subT1)
#    print("subT2 = ", subT2)
#    print("\n\n\nA = ", A)
#    print("\nB = ", B)

    k= len(A)
    print(f"\n\n===> k = {k}")
    print(f"\n===> number of legs (orthants) = k+1 = {k+1}")

    lambda_limits = [0]
    for i in range(k):
        A_i=[T1_edge_dict[str(e)] for e in A[i]]
        B_i=[T2_edge_dict[str(e)] for e in B[i]]
        lambda_limits.append( np.linalg.norm(A_i)/(np.linalg.norm(A_i) + np.linalg.norm(B_i)))
    lambda_limits.append(1)
    lambda_limits = [round(elem,precision) for elem in lambda_limits ]
#    print( "\n\nlambda_limits :", lambda_limits)

    epsilon=[A]
    for i in range(1,k):
        epsilon.append(list(itertools.chain.from_iterable([B[0:i] , A[i:k]])))
    epsilon.append(B)
    print(f"\n===> lambda_limits = {lambda_limits}")
    return(lambda_limits, epsilon)







def pathtree_edges(num, T1, T2, T1_edge_dict, T2_edge_dict,lamda, lambda_limits, epsilon):
    subT1 = T1[num]
    subT2 = T2[num]
    A = subT1[-1]
    B = subT2[-1]
    A_flatten = list(itertools.chain.from_iterable(A))  # "itertools.chain.from_iterable" returns a flattened iterable containing all the elements of the input iterable (hear, it remove the biggest sublists brackets inside a nested list).
    B_flatten = list(itertools.chain.from_iterable(B))
#    print("\n\n\nsubT1 = ", subT1)
#    print("subT2 = ", subT2)
    print(f"\n\n===> A = {A}")
    print(f"\n===> A_flatten = {A_flatten}")
    print(f"\n===> B = {B}")
    print(f"\n===> B_flatten = {B_flatten}")

    
    k= len(A)
#    print(f"\nk = {k} ")
    flatten_epsilon = [ list(itertools.chain.from_iterable(item)) for item in epsilon]
    print(f"\n\n\n===> epsilon :\n {epsilon}")
    print(f"\n===> flatten_epsilon :\n {flatten_epsilon}")

    i = [j for j in range(k+1) if lambda_limits[j]<= lamda <=lambda_limits[j+1]][0]    # i in [0:k]----> number of leg that lamda is there
    print(f"\n\n\n===> for lambda {lamda}: \n i={i} \nsubtree is on the leg(i={i})= {flatten_epsilon[i]}  \n(NOTE: leg index starting from zero)\n\n")
    
    EdgeLength_i=[]
    if i<(k):
        if lamda < lambda_limits[i+1]:
            print("\n***********  NOTE:  Not on the border  ***********" )
            for  edge in flatten_epsilon[i]:
                print(f"\n\nFor edge in flatten_epsilon[i={i}] -----> edge = {edge}")
                if edge in A_flatten:
                    j = next(i for i, v in enumerate(A) if edge in v)
                    print(f"edge in  A_j = A_{j}")
                    print("A[j] = ", A[j])
                    print("B[j] = ", B[j])
                    Aj_norm = np.linalg.norm([T1_edge_dict[str(e)] for e in A[j]])
                    print("Aj_norm = ", Aj_norm)
                    Bj_norm = np.linalg.norm([T2_edge_dict[str(e)] for e in B[j]])
                    print("Bj_norm = ", Bj_norm)
        #            EdgeLength_i.append(format(  (((1-lamda)*Aj_norm-lamda*Bj_norm)/Aj_norm)*(T1_edge_dict[str(edge)])   , '.5f') )
                    EdgeLength_i.append( (((1-lamda)*Aj_norm-lamda*Bj_norm)/Aj_norm)*(T1_edge_dict[str(edge)])  )
                    print(f"new edge_length = {(((1-lamda)*Aj_norm-lamda*Bj_norm)/Aj_norm)*(T1_edge_dict[str(edge)])}")
                if edge in B_flatten:
                    j = next(i for i, v in enumerate(B) if edge in v)    #"j" : the index of B_j of B that edge is inside that
                    print(f"edge in  B_j = B_{j}")
                    print("A[j] = ", A[j])
                    print("B[j] = ", B[j])
                    Aj_norm = np.linalg.norm([T1_edge_dict[str(e)] for e in A[j]])
                    print("Aj_norm = ", Aj_norm)
                    Bj_norm = np.linalg.norm([T2_edge_dict[str(e)] for e in B[j]])
                    print("Bj_norm = ", Bj_norm)
        #            EdgeLength_i.append(format(  ((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])   , '.5f') )
                    EdgeLength_i.append( ((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])  )
                    print(f"new edge_length = {((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])}")
        #    EdgeLength_i = [eval(s.rstrip()) for s in EdgeLength_i]       # "eval" to convert string representation of list to a list

        if lamda == lambda_limits[i+1]:
            print("\n*********** NOTE:  On the border  ***********")
            border_ZeroEdges = [e for e in flatten_epsilon[i] if e not in flatten_epsilon[i+1]]
            print(f"\nborder_ZeroEdges = {border_ZeroEdges}")
            for  edge in flatten_epsilon[i]:
                print(f"\n\nFor edge in flatten_epsilon[i={i}] -----> edge = {edge}")
                if edge in border_ZeroEdges:
                    EdgeLength_i.append(0.0)
                else:
                    if edge in A_flatten:
                        j = next(i for i, v in enumerate(A) if edge in v)
                        print(f"edge in  A_j = A_{j}")
                        print("A[j] = ", A[j])
                        print("B[j] = ", B[j])
                        Aj_norm = np.linalg.norm([T1_edge_dict[str(e)] for e in A[j]])
                        print("Aj_norm = ", Aj_norm)
                        Bj_norm = np.linalg.norm([T2_edge_dict[str(e)] for e in B[j]])
                        print("Bj_norm = ", Bj_norm)
                        #            EdgeLength_i.append(format(  (((1-lamda)*Aj_norm-lamda*Bj_norm)/Aj_norm)*(T1_edge_dict[str(edge)])   , '.5f') )
                        EdgeLength_i.append( (((1-lamda)*Aj_norm-lamda*Bj_norm)/Aj_norm)*(T1_edge_dict[str(edge)])  )
                        print(f"new edge_length = {(((1-lamda)*Aj_norm-lamda*Bj_norm)/Aj_norm)*(T1_edge_dict[str(edge)])}")
                    if edge in B_flatten:
                        j = next(i for i, v in enumerate(B) if edge in v)    #"j" : the index of B_j of B that edge is inside that
                        print(f"edge in  B_j = B_{j}")
                        print("A[j] = ", A[j])
                        print("B[j] = ", B[j])
                        Aj_norm = np.linalg.norm([T1_edge_dict[str(e)] for e in A[j]])
                        print("Aj_norm = ", Aj_norm)
                        Bj_norm = np.linalg.norm([T2_edge_dict[str(e)] for e in B[j]])
                        print("Bj_norm = ", Bj_norm)
                        #            EdgeLength_i.append(format(  ((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])   , '.5f') )
                        EdgeLength_i.append( ((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])  )
                        print(f"new edge_length = {((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])}")
            #    EdgeLength_i = [eval(s.rstrip()) for s in EdgeLength_i]       # "eval" to convert string representation of list to a list
            
            
            
    if i==k:
        print("\n***********  NOTE:  On the last leg  ***********" )
        for  edge in flatten_epsilon[i]:
            print(f"\n\nFor edge in flatten_epsilon[i={i}] -----> edge = {edge}")
            j = next(i for i, v in enumerate(B) if edge in v)    #"j" : the index of B_j of B that edge is inside that
            print(f"edge in  B_j = B_{j}")
            print("A[j] = ", A[j])
            print("B[j] = ", B[j])
            Aj_norm = np.linalg.norm([T1_edge_dict[str(e)] for e in A[j]])
            print("Aj_norm = ", Aj_norm)
            Bj_norm = np.linalg.norm([T2_edge_dict[str(e)] for e in B[j]])
            print("Bj_norm = ", Bj_norm)
            #            EdgeLength_i.append(format(  ((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])   , '.5f') )
            EdgeLength_i.append( ((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])  )
            print(f"new edge_length = {((lamda*Bj_norm-(1-lamda)*Aj_norm)/Bj_norm)*(T2_edge_dict[str(edge)])}")



    
    EdgeLength_i = [round(elem,precision) for elem in EdgeLength_i ]
    print(f"\nEdgeLength_i ={EdgeLength_i}" )      #TEST
    edited_subtree = copy.deepcopy(subT1)     #"deepcopy" to make sure the changes in copy does not effect origional list
    edited_subtree[1][1:] = flatten_epsilon[i]
    
    edited_subtree[3][0] = round((1-lamda)*(subT1[3][0])  + lamda*(subT2[3][0])  , precision)  #length of each common edge " (1-lambda)*e_T +lambda*e_T' "
    edited_subtree[3][1:] = EdgeLength_i         # length of edges(disjoint edges) in the new orthant

    edited_subtree[2] = list( (1-lamda)*(np.array(subT1[2]))  + lamda*(np.array(subT2[2]) ) )  # length of tips(leaves) " (1-lambda)*e_T +lambda*e_T' "
    edited_subtree[2] = [round(elem,precision) for elem in edited_subtree[2] ]
    print("\n\nstarting subtree :\n", subT1)
    print("ending subtree :\n", subT2)
    print("\n\nEditted starting subtree :\n", edited_subtree)

    return(edited_subtree)     #The subtree number "num" in T1 should be replaced with this "edited_subtree" for the given "lamba".










if __name__ == "__main__":
    print("Standalone test")
    if len(sys.argv)<2:
        printf("needs two arguments: treelist and terminallist")
        sys.exit()
    treelist = sys.argv[1]
    terminallist = sys.argv[2]
    Results = subtrees(treelist,terminallist)
    print(Results)
    print(f'\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')



