#!/usr/bin/env python
#Tara Khodaei and Peter Beerli, Tallahassee 2021

# main code for the pathtrees project either check in the parser() section or
# then execute python pathtrees.py --help to learn more
# this project is licensed to you under the MIT opensource license

import sys
import os

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent
sys.path.append(str(file.parent))
current = os.getcwd()


import pathtrees.pathtrees as pt
import pathtrees.likelihood as like
import pathtrees.phylip as ph
import pathtrees.tree as tree
import pathtrees.MDS as plo
import pathtrees.wrf as wrf
import pathtrees.splittree as split
import pathtrees.bifurcating as bifurcating
import pathtrees.optimize as optimize

import numpy as np
import time


#MYJAVA = '/opt/homebrew/Cellar/openjdk/17.0.1_1/bin/java'
MYJAVA =  '/usr/bin/java'
GTPJAR = 'gtp_211101.jar'
GTPTREELIST = 'gtptreelist' # a pair of trees formed from the master treelist
GTPTERMINALLIST = 'terminal_output_gtp'  #GTP terminal output
GTPOUTPUT = 'output.txt' #GTP output file , check later in the source!
GTP = os.path.join(parent, 'pathtrees','gtp')
PAUPTREE = os.path.join(current, 'paup_tree')
PAUPMAP = os.path.join(current, 'PAUP_MAP')     #First Data: comparing with PAUP and MAP
PAUPRXML = os.path.join(current, 'PAUP_RXML_bropt')       #Second Data: comparing with PAUP and RAxML
USER_TREES = os.path.join(current, 'usertrees')
HUGE =100000000


def create_treepair(ti,tj):
    f = open(GTPTREELIST,'w')
    f.write(ti.strip())
    f.write("\n")
    f.write(tj.strip())
    f.write('\n')

def nonzero_lengths(TreeList):
    treelist_new = []
    for t in TreeList:
        t1 = t.replace(':0.0,', ':0.00000001,')
        t2 = t1.replace(':0.0)', ':0.00000001)')
        t3 = t2.replace(':0,', ':0.00000001,')
        t4 = t3.replace(':0)', ':0.00000001)')
        treelist_new.append(t4)
    return treelist_new

def run_gtp(gtptreelist,gtpterminallist,gtpoutput):
    os.system(f"cd  {GTP}; {MYJAVA} -jar {GTPJAR} -v -o {gtpoutput} {gtptreelist} > {gtpterminallist}")

def masterpathtrees(treelist):
    global GTPTERMINALLIST,GTPOUTPUT,GTPTREELIST
    allpathtrees = []
    GTPTERMINALLIST = os.path.join(current,outputdir[it],'terminal_output_gtp')  #GTP terminal output
    GTPOUTPUT = os.path.join(current,outputdir[it],'output.txt')  #GTP output file
    GTPTREELIST = os.path.join(current,outputdir[it],'gtptreelist')  #GTP treelist
    if DEBUG:
        print(f"masterpathtrees: {len(treelist)} trees")
        print(f"gtptreelist={GTPTREELIST}")
        print(f"gtpterminallist={GTPTERMINALLIST}")
    for i,ti in enumerate(treelist):
        for j,tj in enumerate(treelist):
            if j<=i:
                continue
            create_treepair(ti,tj) #this writes into a file GTPTREELIST
            run_gtp(GTPTREELIST, GTPTERMINALLIST, GTPOUTPUT)
            mypathtrees = pt.internalpathtrees(GTPTREELIST, GTPTERMINALLIST, NUMPATHTREES)
            allpathtrees.extend(mypathtrees)
    return [a.strip() for a in allpathtrees]


def likelihoods(trees,sequences):
    likelihood_values=[]
    for i,newtree in enumerate(trees):
        t = tree.Tree()
        t.myread(newtree,t.root)
        t.insertSequence(t.root,labels,sequences)
        
        #setup mutation model, the default for tree is JukesCantor, so these two steps are not really necessary
        Q, basefreqs = like.JukesCantor()
        t.setParameters(Q,basefreqs)
        t.likelihood()
        likelihood_values.append(t.lnL)
    return likelihood_values


def store_results(outputdir,filename,the_list):
    completename = os. path. join(outputdir, filename)
    np.savetxt (completename, the_list,  fmt='%s')

def myparser():
    import argparse
    parser = argparse.ArgumentParser(description='Create a geodesic path between all trees in the treelist, a treefile and a sequence data file are mandatory, if the the option -r N is used then the treefile (which can be empty) will be augemented with N random trees and the pathtree method is then run on those trees')
    parser.add_argument('STARTTREES', 
                        help='mandatory input file that holds a set of trees in Newick format')
    parser.add_argument('DATAFILE', 
                        help='mandatory input file that holds a sequence data set in PHYLIP format')
    parser.add_argument('-o','--output', dest='outputdir', #action='store_const', #const='outputdir',
                        default='output',
                        help='directory that holds the output files')
    parser.add_argument('-v','--verbose', action='store_true',
                        default=None, #const='keep_intermediate',
                        help='Do not remove the intermediate files generated by GPT')
    parser.add_argument('-p','--plot',dest='plotfile',
                        default=None, action='store',
                        help='Create an MDS plot from the generated distances')
    parser.add_argument('-n','--np', '--numpathtrees', dest='NUMPATHTREES',
                        default=10, action='store',
                        help='Number of trees along the path between two initial trees')
    parser.add_argument('-b','--best', '--numbesttrees', dest='NUMBESTTREES',
                        default=str(HUGE), action='store',
                        help='Number of trees selected from the best likliehood trees for the next round of refinement')
    parser.add_argument('-r','--randomtrees', dest='num_random_trees',
                        default=0, action='store',type=int,
                        help='Generate num_random_trees rooted trees using the sequence data individual names.')
    parser.add_argument('-g','--outgroup', dest='outgroup',
                        default=None, action='store',
                        help='Forces an outgroup when generating random trees.')
    parser.add_argument('-i','--iterate', dest='num_iterations',
                        default=1, action='store',type=int,
                        help='Takes the trees, generates the pathtrees, then picks the 10 best trees and reruns pathtrees, this will add an iteration number to the outputdir, and also adds iteration to the plotting.')
    parser.add_argument('-e','--extended', dest='phyliptype',
                        default=None, action='store_true',
                        help='If the phylip dataset is in the extended format, use this.')
    parser.add_argument('-hull','--convex_hull', dest='convex_hull',
                        default=None, action='store_true',
                        help='Extracts the convex hull of input sample trees and considers them as starting trees in the first iteration to generate pairwise pathtrees. If false, it directly considers input sample trees as starting trees in the first iteration to generate pairwise pathtrees')
    parser.add_argument('-gtp','--gtp_distance', dest='gtp',
                        default=None, action='store_true',
                        help='Use GTP derived geodesic distance for MDS plotting [slower], if false use weighted Robinson-Foulds distance for MDS plotting [faster]')
    parser.add_argument('-nel','--neldermead', dest='nel',
                        default=None, action='store_true',
                        help='Use Nelder–Mead optimization method to optimize branchlengths [slower], if false use Newton-Raphson to optimize branchlengths [fast]')
    parser.add_argument('-c','--compare_trees', dest='compare_trees',
                        default=None, action='store',type=str,
                        help='String "D1" considers the first dataset (D_1) with two trees to be compared (PAUP and MAP) with the best tree of PATHTREES, string "D2" considers the second dataset (D_2) with two trees to be compared (PAUP and RAxML) with the best tree of PATHTREES, string "usertrees" considers usertrees to be compared with the best tree of PATHTREES, otherwise it considers nothing to be compared')
    parser.add_argument('-interp','--interpolate', dest='interpolation',
                        default=None, action='store',
                        help='Use interpolation scipy.interpolate.griddata for interpolation [more overshooting], or use scipy.interpolate.Rbf [less overshooting]. String "rbf" considers scipy.interpolate.Rbf, Radial basis function (RBF) thin-plate spline interpolation, with default smoothness=1e-10. String "rbf,s_value", for example "rbf,0.0001", considers scipy.interpolate.Rbf with smoothness= s_value= 0.0001. String "cubic" considers scipy.interpolate.griddata, cubic spline interpolation. Otherwise, with None interpolation, it considers default scipy.interpolate.Rbf with smoothness=1e-10 ')
    parser.add_argument('-valid','--validatioon', dest='validation_mds',
                        default=None, action='store_true',
                        help=' Validates the MDS plots by computing correlation measures Pearson r,  Spearman rho, and Kendall Tau between the original distances and the MDS distances, and a plot showing the real distances VS MDS distances')  #Dec27.2022
                            

    args = parser.parse_args()
    return args

    
if __name__ == "__main__":

    DEBUG = False
    Optimize_Randoms= False
        
    args = myparser() # parses the commandline arguments
    start_trees = args.STARTTREES
    datafile = args.DATAFILE
    outputdir = args.outputdir
    keep = args.verbose == True
    num_random_trees = args.num_random_trees
    outgroup = args.outgroup
    num_iterations = args.num_iterations+1
    plotfile = args.plotfile
    convex_hull = args.convex_hull
    gtp_dist = args.gtp
    nel = args.nel
    validation_mds = args.validation_mds


    interpolation = args.interpolation
    if interpolation is not None:
        my_interpolation = [item for item in interpolation.split(',')]
        interpolation_type = my_interpolation[0]
        if interpolation_type == "rbf":
            if len(my_interpolation)>1:
                smoothness = float(my_interpolation[1])
            else:
                smoothness = 1e-10
        else:
            interpolation_type == "cubic"
            smoothness = None
    else:
        interpolation_type = "rbf"
        smoothness = 1e-10
        
    if DEBUG:
        print(f"interpolation_type = {interpolation_type}")
        print(f"smoothness = {smoothness}")

    

    compare_trees = args.compare_trees
    if compare_trees == "D1":     #dataset1 : PAUP and MAP to be compared
        paup_tree = False
        paup_MAP = True
        paup_RXML = False
        usertrees = False
        if DEBUG:
            print(f"compare_trees_list = {[paup_tree, paup_MAP, paup_RXML, usertrees]}")
            
    elif compare_trees == "D2":     #dataset2: PAUP and RAxML to be compared
        paup_tree = False
        paup_MAP = False
        paup_RXML = True
        usertrees = False
        if DEBUG:
            print(f"compare_trees_list = {[paup_tree, paup_MAP, paup_RXML, usertrees]}")
            
    elif compare_trees is not None and compare_trees != "D1" and compare_trees != "D2":           #user extra-trees to be compared
        paup_tree = False
        paup_MAP = False
        paup_RXML = False
        usertrees = True
        if DEBUG:
            print(f"compare_trees_list = {[paup_tree, paup_MAP, paup_RXML, usertrees]}")
            
    elif compare_trees is None:     #noting to be compared
        paup_tree = False
        paup_MAP = False
        paup_RXML = False
        usertrees = False
        if DEBUG:
            print(f"compare_trees_list = {[paup_tree, paup_MAP, paup_RXML, usertrees]}")
    compare_trees_list = [paup_tree, paup_MAP, paup_RXML, usertrees]
    
    
    phyliptype = args.phyliptype
    if phyliptype:
        ttype = 'EXTENDED'
        filetype = 'RelPHYLIP'
    else:
        ttype = 'STANDARD'
        filetype = 'PHYLIP'

            
    if num_iterations<=1:
        os.system(f'mkdir -p {outputdir}')
        outputdir = [outputdir]
        if plotfile != None:
            plotfile2 = "MDS_"+plotfile+".pdf"
    else:
        plotfile2 = []
        o = outputdir
        outputdir=[]
        for it in range(1,num_iterations):
            os.system(f'mkdir -p {o}{it}')
            outputdir.append(f'{o}{it}')
            plotfile2.append(f"MDS_iter{it}.pdf")


    
    STEPSIZE = 1  # perhaps this should be part of options
    labels, sequences, variable_sites = ph.readData(datafile, ttype)
    
    
    if convex_hull:
        with open(start_trees,'r') as f:
            sampletrees = [line.strip() for line in f]
        store_results(outputdir[0],'sampletrees', sampletrees)

        GTPOUTPUT = os.path.join(current,outputdir[0],'output.txt')
        new_sampletrees = os.path.join(current, outputdir[0], 'sampletrees')
        if gtp_dist:
            run_gtp(new_sampletrees, GTPTERMINALLIST, GTPOUTPUT)
            distances = plo.read_GTP_distances(len(sampletrees),GTPOUTPUT)
        else:
            distances = wrf.RF_distances(len(sampletrees), new_sampletrees, type="weighted")
        
        Likelihoods_sampletrees = likelihoods(sampletrees,sequences)
                    
        iter_num=1
        Boundary_Trees = plo.boundary_convexhull(distances,Likelihoods_sampletrees,sampletrees, iter_num)
        store_results(outputdir[0],'Boundary_Trees',Boundary_Trees)
        np.savetxt ('Boundary_Trees',Boundary_Trees, fmt='%s')
        StartTrees = Boundary_Trees
        sys.exit()
    else:
        with open(start_trees,'r') as f:
            StartTrees = [line.strip() for line in f]
    

    NUMPATHTREES_list = list(map(int, args.NUMPATHTREES.split(',')))
    NUMBESTTREES_list = list(map(int, args.NUMBESTTREES.split(',')))
    NUMPATHTREES = NUMPATHTREES_list[0]
    NUMBESTTREES = NUMBESTTREES_list[0]

    #==========================  Random Trees  ============================
    if num_random_trees>0:
        from numpy.random import default_rng
        totaltreelength = ph.guess_totaltreelength(variable_sites)
        rng = default_rng()
        randomtrees = [tree.generate_random_tree(labels, rng.uniform(0.2,100)*totaltreelength, outgroup) for _ in range(num_random_trees)]
        StartTrees =  randomtrees
    #======================================================================

    if paup_tree:        #just paup tree
        with open(PAUPTREE,'r') as myfile:
            T_opt = myfile.readlines()
            paup = bifurcating.bifurcating_newick(T_opt)

    elif paup_MAP:       #paup + MAP trees
        with open(PAUPMAP,'r') as myfile:
            T_opt = myfile.readlines()
            paup_MAP = bifurcating.bifurcating_newick(T_opt)

    elif paup_RXML:      #paup + RXML trees (bropt: branches optimized)
        with open(PAUPRXML,'r') as myfile:
            T_opt = myfile.readlines()
            paup_RXML = bifurcating.bifurcating_newick(T_opt)

    elif usertrees:     #user trees
        with open(USER_TREES,'r') as myfile:
            T_opt = myfile.readlines()
            usertrees = bifurcating.bifurcating_newick(T_opt)

    
    for it1 in range(1,num_iterations):
        if it1==1:
            print(f'\n\n============================  iteration {it1}  ================================')
        it = it1-1
        
        if DEBUG:
            print(f'\nGenerating pathtrees through tree space...')
        
        
        if Optimize_Randoms and it==0:
            print(f'\nOptimizing random starttrees...\n')
            tic = time.perf_counter()
            optimized_starttrees=[]
            for j,tree_num in enumerate(StartTrees):
                mytree = tree.Tree()
                mytree.myread(tree_num,mytree.root)
                labels, sequences, variable_sites = ph.readData(datafile)
                mytree.insertSequence(mytree.root,labels,sequences)
                optnewick = mytree.paupoptimize(datafile, filetype="RelPHYLIP")
#                optnewick = mytree.paupoptimize(datafile, filetype="PHYLIP")
                optimized_starttrees.append(optnewick)
            toc = time.perf_counter()
            timeing = toc - tic
            if DEBUG:
                print(f"\nTime of optimizing starttrees = {timeing}")
            store_results(outputdir[it],'optimized_starttrees',optimized_starttrees)
            # make bifurcating and nonzero since we optimized trees that may be not biforcating now in the second iteration
            T_bifurcate = bifurcating.bifurcating_newick(optimized_starttrees)
            optimized_starttrees = nonzero_lengths(optimized_starttrees)
            if DEBUG:
                store_results(outputdir[it],'bifurcating_newick',optimized_starttrees)
            StartTrees = optimized_starttrees
            print(f"\nlength of random_and_optimize = len(StartTrees) =  {len(StartTrees)}")
    
    
        store_results(outputdir[it],'starttrees',StartTrees)
        print(f"\n    length of StartTrees  = {len(StartTrees)}")
                
        
        print(f'\n\n>>> Generating {NUMPATHTREES-2}-pathtree(s) for each pair of StartTrees ...')
        tic = time.perf_counter()
        Pathtrees = masterpathtrees(StartTrees)
        store_results(outputdir[it],'pathtrees',Pathtrees)
        print(f"    length of Pathtrees = {len(Pathtrees)}")
        toc = time.perf_counter()
        timeing = toc - tic
        if DEBUG:
            print(f"Time of generating pathtrees = {timeing}")
        Treelist= StartTrees+Pathtrees
        print(f"    length of treelist = len(StartTrees) + len(Pathtrees) = {len(StartTrees)} + {len(Pathtrees)} = {len(Treelist)}")
        
        if DEBUG:
            print(f'\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Likelihood ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        print(f'\n\n>>> Calculating likelihoods ...')
        tic = time.perf_counter()
        Likelihoods = likelihoods(Treelist,sequences)
        idx = plo.best_likelihoods(Likelihoods,NUMBESTTREES)
        toc = time.perf_counter()
        timeing = toc - tic
        if DEBUG:
            print(f"\nTime of  calculating {len(Treelist)}-trees Likelihood = {timeing}")



        if DEBUG:
            print(f'\n~~~~~~~~~~~~~  Topologies of best {len(idx)}-trees using unweighted-RF ~~~~~~~~~~~\n')
        print(f'\n\n>>> Calculating topologies of best {len(idx)}-trees using unweighted-RF ...')
        tic = time.perf_counter()
        BestTrees1=[Treelist[i] for i in idx]
        BestTrees= bifurcating.bifur_to_mulfur_newick(BestTrees1)
        
        store_results(outputdir[it],'BestTrees',BestTrees)
        dict_idx = dict(zip(range(0, len(idx)) , idx))
        best_treelist = os.path.join(outputdir[it], 'BestTrees')
        best_distances = wrf.RF_distances(len(BestTrees), best_treelist, type="unweighted")
        
        L= []
        for i in range(len(best_distances)):
            l=[]
            for j in range(len(best_distances)):
                if best_distances[i,j]==0:
                    l.append(j)
            L.append(l)
        Topologies = []
        for i in L:
            if i not in Topologies:
                Topologies.append(i)

        best_topo_like_idx = []
        for i in range(len(Topologies)):
            topo = Topologies[i]
            topo_idx =[dict_idx[j] for j in topo ]
            Topologies[i] = topo_idx
            Like = [Likelihoods[l] for l in topo_idx]
            sort_index= sorted(range(len(Like)), key=lambda k: Like[k])
            best_topo_idx = topo_idx[sort_index[-1]]
            best_topo_like_idx.append(best_topo_idx)
        
        print(f"    Number of topologies = {len(Topologies)}")
        if DEBUG:
            print(f"Topologies = {Topologies}\n")
            print(f"best_topo_like_idx = {best_topo_like_idx}")
        
        toc = time.perf_counter()
        timeing = toc - tic
        if DEBUG:
            print(f"Time of  generating topologies = {timeing}")
        
        #length of pathtrees before optimization : To do histogram analysis:
        pathtrees_len_before =[]
        for k in range(len(Pathtrees)):
            p = tree.Node()
            p.name = 'root'
            mytree = tree.Tree()
            mytree.root=p
            mytree.myread(Pathtrees[k],p)
            treelen = mytree.tree_len
            pathtrees_len_before.append(treelen)
        pathtrees_len_before_opt =[]
        pathtrees_len_after_opt =[]
        
  
  
        if DEBUG:
            print(f'\n\n~~~~~~~~~~~  Optimizing one tree from each topology(highest loglike tree)  ~~~~~~~~~~\n')
        print(f'\n\n>>> Optimizing one tree from each topology ...')
        tic = time.perf_counter()
        optimized_BestTrees=[]
        for j,num in enumerate(best_topo_like_idx):
            tree_num = Treelist[num]
            mytree = tree.Tree()
            mytree.myread(tree_num,mytree.root)
            
            treelen1 = mytree.tree_len
            pathtrees_len_before_opt.append(treelen1)
            
            mytree.insertSequence(mytree.root,labels,sequences)

            #paup optimize / Nelder-Mead optimize
            if nel:
                x,fval,iterations, tree1 = optimize.minimize_neldermead(mytree, maxiter=None, maxiter_multiplier=200, initial_simplex=None, xatol=1e-4, fatol=1e-4, adaptive=False, bounds=[0,1e6])
                optnewick = tree1.treeprintvar().rstrip()   # assumes you have the new tree.py & rstrip() to remove \n from end of each new tree
            else:
                optnewick = mytree.paupoptimize(datafile, filetype="RelPHYLIP")

            mytree2 = tree.Tree()
            mytree2.myread(optnewick,mytree2.root)
            treelen2 = mytree2.tree_len
            pathtrees_len_after_opt.append(treelen2)
            optimized_BestTrees.append(optnewick)
        toc = time.perf_counter()
        timeing = toc - tic
        if DEBUG:
            print(f"Time of optimizing = {timeing}")
        
        if DEBUG:    #To do histogram analysis
            print(f"\npathtrees_len_before_opt = {pathtrees_len_before_opt}")
            print(f"\npathtrees_len_after_opt = {pathtrees_len_after_opt}")

        
        Likelihoods_optimized_BestTrees = likelihoods(optimized_BestTrees,sequences)
        sort_index_opt = sorted(range(len(Likelihoods_optimized_BestTrees)), key=lambda k: Likelihoods_optimized_BestTrees[k])
        Like_opt = [Likelihoods_optimized_BestTrees[i] for i in sort_index_opt]
        BestTrees_opt = [optimized_BestTrees[i] for i in sort_index_opt]
        Topologies_opt = [Topologies[i] for i in sort_index_opt]

        Treelist += BestTrees_opt
        print(f"    length of treelist = len(StartTrees) + len(Pathtrees) + len(Optimized Trees) = {len(StartTrees)} + {len(Pathtrees)} + {len(BestTrees_opt)} =  {len(Treelist)}")
    
        Likelihoods += Like_opt
        
        

        with open(os.path.join(outputdir[it], "PATHTREES_optimal"), "w") as myfile:
            myfile.write("Tree:\n" + "".join(Treelist[-1]) + "\n")
            myfile.write("\nLog-Likelihood:\n" + "".join(str(Likelihoods[-1])) + "\n")
        

                
        if paup_tree:
            print(f'\n\n>>> Adding some external trees to treelist to compare ...')
            Treelist += paup
            print(f"    length of treelist  after adding  1-tree (PAUP) = {len(Treelist)}\n\n")
            Likelihood_PAUP = likelihoods(paup,sequences)
            Likelihoods += Likelihood_PAUP
            if DEBUG:
                print(f"length of Likelihoods after adding 1-tree (PAUP) ---->  {len(Likelihoods)}")
        elif paup_MAP:
            print(f'\n\n>>> Adding some external trees to treelist to compare ...')
            Treelist += paup_MAP
            print(f"    length of treelist  after adding  2-trees (PAUP & MAP) = {len(Treelist)}\n\n")
            Likelihood_PAUP_MAP = likelihoods(paup_MAP,sequences)
            Likelihoods += Likelihood_PAUP_MAP
            if DEBUG:
                print(f"length of Likelihoods after adding 2-trees (PAUP & MAP) ---->  {len(Likelihoods)}")
        elif paup_RXML:
            print(f'\n\n>>> Adding some external trees to treelist to compare ...')
            Treelist += paup_RXML
            print(f"    length of treelist  after adding  2-trees (PAUP & RAxML) = {len(Treelist)}\n\n")
            Likelihood_PAUP_RXML = likelihoods(paup_RXML,sequences)
            Likelihoods += Likelihood_PAUP_RXML
            if DEBUG:
                print(f"length of Likelihoods after adding 2-trees (PAUP & RAxML) ---->  {len(Likelihoods)}")
        elif usertrees:
            print(f'\n\n>>> Adding some external trees to treelist to compare ...')
            Treelist += usertrees
            print(f"    length of treelist  after adding {len(usertrees)}-trees (usertrees) = {len(Treelist)}\n\n")
            Likelihood_usertrees = likelihoods(usertrees,sequences)
            Likelihoods += Likelihood_usertrees
            if DEBUG:
                print(f"length of Likelihoods after adding {len(usertrees)}-trees (usertrees) ---->  {len(Likelihoods)}")


        store_results(outputdir[it],'treelist',Treelist)
        store_results(outputdir[it],'likelihoods',Likelihoods)
        
        if DEBUG:
            store_results(outputdir[it],'Topologies_opt',Topologies_opt)
            store_results(outputdir[it],'optimized_BestTrees',BestTrees_opt)
        

        if plotfile != None:
            if DEBUG:
                print(f'\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MDS plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            print(f'>>> Generating 2D and 3D plots of generated tree landscape ...\n')
            tic3 = time.perf_counter()
            newtreelist = os.path.join(current,outputdir[it], 'treelist')
            if not keep:
                os.system(f'rm {GTPTERMINALLIST}')
                os.system(f'rm {GTPTREELIST}')
            if plotfile != None:
                if paup_tree:
                    bestlike = plo.best_likelihoods(Likelihoods[:-(len(BestTrees_opt)+1)],NUMBESTTREES)   # minus optimizaed trees and 1PAUP
                elif paup_MAP:
                    bestlike = plo.best_likelihoods(Likelihoods[:-(len(BestTrees_opt)+2)],NUMBESTTREES)   # minus optimizaed trees and 1PAUP&1MAP
                elif paup_RXML:
                    bestlike = plo.best_likelihoods(Likelihoods[:-(len(BestTrees_opt)+2)],NUMBESTTREES)    # minus optimizaed trees and 1PAUP&1RXML
                elif usertrees:
                    bestlike = plo.best_likelihoods(Likelihoods[:-(len(BestTrees_opt)+len(usertrees))],NUMBESTTREES)    # minus optimizaed trees and 1PAUP&1RXML
                else:
                    bestlike = plo.best_likelihoods(Likelihoods[:-(len(BestTrees_opt))],NUMBESTTREES)   # minus optimizaed trees
                n = len(Treelist)
                
                #RF_distance  / GTP_distance:
                if gtp_dist:
                    run_gtp(newtreelist, GTPTERMINALLIST, GTPOUTPUT)
                    distances = plo.read_GTP_distances(n,GTPOUTPUT)
                    if DEBUG:
                        store_results(outputdir[it],'GTP_distances',distances)
                else:
                    distances = wrf.RF_distances(n, newtreelist, type="weighted")
                    if DEBUG:
                        store_results(outputdir[it],'RF_distances',distances)
                

                if DEBUG:
                    plo.plot_MDS(plotfile, N, n, distances, Likelihoods, bestlike, Treelist, Pathtrees)
                    
                if interpolation_type == "cubic":
                    plo.interpolate_grid(it, plotfile2[it], distances,Likelihoods, bestlike,Treelist, StartTrees,BestTrees_opt, Topologies_opt,NUMPATHTREES, compare_trees_list , compare_trees, validation_mds)
                elif interpolation_type == "rbf":
                    plo.interpolate_rbf(it, plotfile2[it], distances,Likelihoods, bestlike,Treelist, StartTrees,BestTrees_opt, Topologies_opt,NUMPATHTREES, compare_trees_list , compare_trees, smoothness, validation_mds)


        print(f'\nNOTE :')
        print(f'    find the following outputs of iteration{it1} in the folder "output{it1}":')
        print(f'    - "starttrees"  includes the starting trees')
        print(f'    - "pathtrees"  includes all pathtrees generated by PATHTREES using starttrees')
        print(f'    - "treelist"  shows all trees in the order of starttrees + pathtrees + optimized trees + external trees')
        print(f'    - "likelihoods" shows the log-likeligood values of treelist')
        print(f'    - "PATHTREES_optimal" includes the highest likelihood tree found by PATHTREES and the corresponding log-likelihood value')
        
        print(f'\n    find the following plots in the directory:')
        print(f'    - "MDS_iter{it1}"  shows 2D and 3D plot of generated tree lanscape')
        if it1>1:
            print(f'    - "Boundary_iter{it1}"  displays the boundary trees of optimized trees from previous iteration')
        if validation_mds:
            print(f'    - "ShepardDiagram_iter{it1}"  is the shepard diagram showing real distances vs MDS distances')
        print(f'\n')
        

        if it1 < num_iterations-1:
            if DEBUG:
                print(f'\n\n~~~~~~~~~~~~~  For next iteration ---> Boundary of optimized trees ~~~~~~~~~~~~~~~~\n')
            tic = time.perf_counter()
            
            # make bifurcating and nonzero since we optimized trees that may be not biforcating now in the second iteration
            T_bifurcate = bifurcating.bifurcating_newick(BestTrees_opt)
            T_nonzero = nonzero_lengths(T_bifurcate)
            
            StartTrees_forboundaries = T_nonzero
            Likelihoods_starttrees =  likelihoods(StartTrees_forboundaries, sequences)
            if DEBUG:
                store_results(outputdir[it1],'BestTrees_opt',BestTrees_opt)
                store_results(outputdir[it1],'T_bifurcate',T_bifurcate)
                store_results(outputdir[it1],'T_nonzero',T_nonzero)
                
            store_results(outputdir[it1],'StartTrees_forboundaries',StartTrees_forboundaries)
            distancefile = os.path.join(outputdir[it1], 'StartTrees_forboundaries')
            distances =  wrf.RF_distances(len(StartTrees_forboundaries), distancefile, type="weighted")
            
            iter_num = it1+1
            print(f'\n============================  iteration {iter_num}  ================================\n')
            print(f'>>> Extracting boundary trees of {len(StartTrees_forboundaries)} optimized trees from previous iteration ...')
            Boundary_Trees = plo.boundary_convexhull(distances,Likelihoods_starttrees,StartTrees_forboundaries, iter_num)
            
            
            toc = time.perf_counter()
            timeing = toc - tic
            if DEBUG:
                store_results(outputdir[it1],'Boundary_Trees',Boundary_Trees)
                print(f"Time of generating BoundaryTrees = {timeing}")
            

            StartTrees = Boundary_Trees
            if len(NUMPATHTREES_list)>it1:
                NUMPATHTREES = NUMPATHTREES_list[it1]
                
            if len(NUMBESTTREES_list)>it1:
                NUMBESTTREES = NUMBESTTREES_list[it1]
            else:
                NUMBESTTREES = HUGE
            

                
                
            








