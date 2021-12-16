#!/usr/bin/env python
#
# main code for the pathtrees project either check in the parser() section or
# then execute python pathtrees.py --help
# to learn more
#
#
# (c) Tara Khodaei and Peter Beerli, Tallahassee 2021
# this project is licensed to you under the MIT opensource license
# 
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

import numpy as np
import time

import dendropy
from dendropy.calculate import treecompare


MYJAVA = '/opt/homebrew/Cellar/openjdk/17.0.1_1/bin/java'
GTPJAR = 'gtp_211101.jar'
GTPTREELIST = 'gtptreelist' # a pair of trees formed from the master treelist
GTPTERMINALLIST = 'terminal_output_gtp'  #GTP terminal output
GTPOUTPUT = 'output.txt' #GTP output file , check later in the source!
NUMPATHTREES = 10  #number of trees in path
GTP = os.path.join(parent, 'pathtrees','gtp')

#def create_treepair(ti,tj,pairtreelist):
#    f = open(pairtreelist,'w')
#    f.write(ti.strip())
#    f.write("\n")
#    f.write(tj.strip())
#    f.write('\n')
#    f.close()

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
    print(f"cd  {GTP}; {MYJAVA} -jar {GTPJAR} -v -o {gtpoutput} {gtptreelist} > {gtpterminallist}",file=sys.stderr)
    os.system(f"cd  {GTP}; {MYJAVA} -jar {GTPJAR} -v -o {gtpoutput} {gtptreelist} > {gtpterminallist}")

def masterpathtrees(treelist): #this is the master treelist
    # loop over treelist:
    allpathtrees = []
    if DEBUG:
        print(f"masterpathtrees: {len(treelist)} trees")
        print(f"gtptreelist={GTPTREELIST}")
        print(f"gtpterminallist={GTPTERMINALLIST}")
    for i,ti in enumerate(treelist):
        for j,tj in enumerate(treelist):
            if j<=i:
                continue
            #form a treelist of the pair
#            create_treepair(ti,tj,GTPTREELIST) #writes to a file GTPTREELIST
            create_treepair(ti,tj) #this writes into a file GTPTREELIST
#            run_gtp(GTPTREELIST, GTPTERMINALLIST, GTPOUTPUT)
            run_gtp(GTPTREELIST, GTPTERMINALLIST)
            mypathtrees = pt.internalpathtrees(GTPTREELIST, GTPTERMINALLIST, NUMPATHTREES)
            allpathtrees.extend(mypathtrees)
    return [a.strip() for a in allpathtrees]



#def likelihoods(trees,sequences, opt=False):
#    #global labels
#    if DEBUG:
#        print("Likelihood:",f"number of trees:{len(trees)}")
#        print("Likelihood:",f"number of sequences:{len(sequences)}")
#        print("Likelihood:",f"optimize:{opt}")
#    likelihood_values=[]
#    newtrees = [] # for opt=True
#    lt = len(trees)
#    for i,newtree in enumerate(trees):
#        t = tree.Tree()
#        t.myread(newtree,t.root)
#        t.insertSequence(t.root,labels,sequences)
#
#        #setup mutation model
#        # the default for tree is JukesCantor,
#        # so these two steps are not really necessary
#        Q, basefreqs = like.JukesCantor()
#        t.setParameters(Q,basefreqs)
#        #calculate likelihood and return it
#        if opt:
#            if NR:
#                t.optimizeNR()
#            else:
#                #t.optimize()
#                pnewick = t.paupoptimize(datafile,filetype)
#                pnewtree = Tree()
#                pnewtree.root.name='root'
#                pnewtree.root.blength=0.0
#                pnewtree.myread(pnewick,newtree.root)
#                pnewtree.insertSequence(pnewtree.root,labels,sequences)
#                pnewtree.likelihood()
#                t = pnewtree
#            if DEBUG:
#                print(f"optimized tree {i} of {lt} with lnL={t.lnL}")
#            likelihood_values.append(t.lnL)
#            with split.Redirectedstdout() as newick:
#                t.myprint(t.root,file=sys.stdout)
#            newtrees.append(str(newick)+';')
#        else:
#            t.likelihood()
#            likelihood_values.append(t.lnL)
#            if DEBUG:
#                print("Likelihood:",f"lnL={t.lnL} {newtree[:50]}")
#    return likelihood_values, newtrees



def likelihoods(trees,sequences):
    likelihood_values=[]
    for i,newtree in enumerate(trees):
        t = tree.Tree()
        t.myread(newtree,t.root)
        t.root.name = 'root'
        t.insertSequence(t.root,labels,sequences)
        
        #setup mutation model
        # the default for tree is JukesCantor,
        # so these two steps are not really necessary
        Q, basefreqs = like.JukesCantor()
        t.setParameters(Q,basefreqs)
        #calculate likelihood and return it
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
    parser.add_argument('-o','--output', dest='outputdir', #action='store_const',
                        #const='outputdir',
                        default='pathtree_outputdir',
                        help='directory that holds the output files')
    parser.add_argument('-v','--verbose', action='store_true',
                        default=None, #const='keep_intermediate',
                        help='Do not remove the intermediate files generated by GPT')
    parser.add_argument('-p','--plot',dest='plotfile',
                        default=None, action='store',
                        help='Create an MDS plot from the generated distances')
    parser.add_argument('-n','--np', '--numpathtrees', dest='NUMPATHTREES',
                        default=10, action='store',type=int,
                        help='Number of trees along the path between two initial trees')
    parser.add_argument('-b','--best', '--numbesttrees', dest='NUMBESTTREES',
                        default=10, action='store',type=int,
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
    parser.add_argument('-bound','--hull', '--boundary', dest='proptype',
                        default=None, action='store_true',
                        help='Start the iteration using a convex hull instead of n best likelihood trees.')
    parser.add_argument('-f','--fast', '--wrf', dest='fast',
                        default=None, action='store_true',
                        help='use weighted Robinson-Foulds distance for MDS plotting [fast], if false use GTP derived geodesic distance [slow]')

    parser.add_argument('-allopt','--alloptimize', '--allopt', dest='allopt',
                        default=False, action='store_true',
                        help='calculates the pathtrees and then finds all optimal branchlengths for each of them')

    parser.add_argument('-opt','--optimize', '--opt', dest='opt',
                        default=False, action='store_true',
                        help='finds optimal branchlengths for the best trees')

    parser.add_argument('-optnr','--optimizenr', '--optnr', dest='NR',
                        default=False, action='store_true',
                        help='finds optimal branchlengths using Newton-Raphson for each of them')

    args = parser.parse_args()
    return args

    
if __name__ == "__main__":
    
#    paup_tree = False    #first data option
#    paup_MAP = True    #first data option
#    paup_tree = True    #second data option
#    paup_MAP = False    #second data option
    paup_tree = False    #No paup
    paup_MAP = False    #No paup&MAp
    
    DEBUG = False
    RANDOM_TREES= False
    Generate_trees = False
    
    RF_distances = True     #: False means  GTP
    paup_optimize = True    #: False means  neldermead
    
    
    args = myparser() # parses the commandline arguments
    start_trees = args.STARTTREES
    datafile = args.DATAFILE
    outputdir = args.outputdir
    keep = args.verbose == True
    num_random_trees = args.num_random_trees
    outgroup = args.outgroup
    num_iterations = args.num_iterations+1
    plotfile = args.plotfile
    
    #~~~~~~~~~~~~~~~~~~~????~~~~~~~~~~~~~~~~~
    fast = args.fast
    allopt = args.allopt
    opt = args.opt
    NR = args.NR
    proptype = args.proptype
    if proptype:
        from scipy.spatial import ConvexHull
    
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
            plotfile2 = "contour_"+plotfile
    else:
        plotfile2 = []
        o = outputdir
        outputdir=[]
        for it in range(1,num_iterations):
            os.system(f'mkdir -p {o}{it}')
            outputdir.append(f'{o}{it}')
            plotfile2.append(f"contour_{it}_{plotfile}")

    NUMPATHTREES = args.NUMPATHTREES
    NUMBESTTREES = args.NUMBESTTREES          #???
    STEPSIZE = 1  # perhaps this should be part of options          #???

    if DEBUG:
        print(args)
        print(datafile)
        print(ttype)

    labels, sequences, variable_sites = ph.readData(datafile, ttype)          #???   CHECK this
#    labels, sequences, variable_sites = ph.readData(datafile)

    ############################  Random Trees  #########################
    if num_random_trees>0:
        from numpy.random import default_rng
        totaltreelength = ph.guess_totaltreelength(variable_sites)
        rng = default_rng()        
        randomtrees = [tree.generate_random_tree(labels, rng.uniform(0.2,100)*totaltreelength, outgroup) for _ in range(num_random_trees)]
        #print(randomtrees)
        #sys.exit()
        with open(start_trees,'a') as f:
            for rt in randomtrees:
                print(rt,file=f)
    #####################################################################

    if paup_tree:     #has_paup_tree
        print(f'\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~ reading paup tree ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        with open("paup_tree",'r') as myfile:      #just PAUP
            T_opt = myfile.readlines()
            paup = bifurcating.bifurcating_newick(T_opt)
#            np.savetxt ("paup", paup,  fmt='%s')
    elif paup_MAP:    #has_paup_Map_tree
        print(f'\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~ reading paup&MAP tree ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        with open("both_optimized.tre",'r') as myfile:      #both PAUP&MAP
            T_opt = myfile.readlines()
            paup_MAP = bifurcating.bifurcating_newick(T_opt)
#            np.savetxt ("paup_MAP", paup_MAP,  fmt='%s')


    print(f'\n\n~~~~~~~~~~~~~~~~~~Calculating paths through tree space ~~~~~~~~~~~~~~~~~~')

#    tictotal = time.perf_counter()

    if Generate_trees:       #clean treelist and choose every dt tree from treelist ( NOTE: origional trees may not be rooted make sure to root them)
        StartTrees = tree.generate_treelist(start_trees, 80)
    else:
        with open(start_trees,'r') as f:      # trees are cleaned(every dt trees and rooted), or our randonm trees
            StartTrees = [line.strip() for line in f]
    print(f"\nlength of StartTrees ---->  {len(StartTrees)}")


    print(f'\n\n\n~~~~~~~~~~~~~~~~~~  TEST : smaller number of start trees  ~~~~~~~~~~~~~~~~~~\n\n')
    idx=np.arange(0,len(StartTrees),1)     #change 1 to other values
    StartTrees = [StartTrees[i] for i in idx]
    print(f"\nlength of StartTrees ---->  {len(StartTrees)}")



    for it1 in range(1,num_iterations):
        print(f'\n\n\n===============================  iteration{it1}  ==============================')
        it = it1-1
        if RANDOM_TREES:
            print(f'\n\n================= Random starting trees ==========================')
            print(f'~~~~~~~~~~~~~~~~~~~~~~~  Optimizing starttrees  ~~~~~~~~~~~~~~~~~~~~~~')
            tic = time.perf_counter()
            
            optimized_starttrees=[]
            for j,tree_num in enumerate(StartTrees):
                mytree = tree.Tree()
                mytree.myread(tree_num,mytree.root)
                labels, sequences, variable_sites = ph.readData(datafile)
                mytree.insertSequence(mytree.root,labels,sequences)
#                optnewick = mytree.paupoptimize(datafile, filetype="RelPHYLIP")
                optnewick = mytree.paupoptimize(datafile, filetype="PHYLIP")
                
                optimized_starttrees.append(optnewick)
            toc = time.perf_counter()
            timeing = toc - tic
            print(f"\n\n\nTime of optimizing starttrees = {timeing}")
            store_results(outputdir[it],'optimized_starttrees',optimized_starttrees)
            
            if it>0 : # make bifurcating and nonzero since we optimized trees that may be not biforcating now in the second iteration
                T_bifurcate = bifurcating.bifurcating_newick(optimized_starttrees)
                optimized_starttrees = nonzero_lengths(optimized_starttrees)
                store_results(outputdir[it],'bifurcating_newick',optimized_starttrees)   #new

            StartTrees = optimized_starttrees
            print(f"\nlength of random_and_optimize ---->  {len(StartTrees)}")
    
        store_results(outputdir[it],'StartTrees',StartTrees)
        print(f"\nlength of StartTrees ---->  {len(StartTrees)}")
        
        
        print(f'\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~ pathtrees ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
#        tic = time.perf_counter()
#        GTPTREELIST = os.path.join(current,outputdir[it],'gtptreelist') # a pair of trees formed from the master treelist
#        GTPTERMINALLIST = os.path.join(current,outputdir[it],'terminal_output_gtp')  #GTP terminal output
#        GTPOUTPUT = os.path.join(current,outputdir[it],'output.txt')  #GTP output file
#        Pathtrees = masterpathtrees(StartTrees)
#        slen = len(StartTrees)
#        Treelist= StartTrees+Pathtrees
#        Likelihoods, newtreelist  = likelihoods(Treelist,sequences,allopt)
#        if allopt:
#            StartTrees = newtreelist[:slen]
#            Pathtrees = newtreelist[slen:]
#            Treelist = newtreelist[:]
#        Likelihoods = [like if like != -np.inf else -10**8 for like in Likelihoods]
#        store_results(outputdir[it],'likelihood',Likelihoods)
#        store_results(outputdir[it],'treelist',Treelist)
#        store_results(outputdir[it],'starttrees',StartTrees)
#        store_results(outputdir[it],'pathtrees',Pathtrees)
#        toc = time.perf_counter()
#        time1 = toc - tic
#        print(f"Time of generating pathtrees results = {time1}")
#        tic2 = time.perf_counter()
#
#        newtreelist = os.path.join(outputdir[it], 'treelist')
#        if not fast:
#            print('Calculate geodesic distance among all pathtrees')
#            run_gtp(newtreelist, GTPTERMINALLIST,GTPOUTPUT)
#            #os.system(f'mv pathtrees/gtp/output.txt {outputdir[it]}/')
#            #if not keep:
#            #    os.system(f'rm {GTPTERMINALLIST}')
#            #    os.system(f'rm {GTPTREELIST}')
#            toc2 = time.perf_counter()
#            time2 = toc2 - tic2
#            print(f"Time of GTP distances of all trees = {time2}")
#
#        bestlike = plo.bestNstep_likelihoods(Likelihoods,NUMBESTTREES,STEPSIZE)
#
#        if opt:
#            bestindex = list(zip(*bestlike))[0]
#            besttrees = np.take(Treelist,bestindex)
#            newbestlikelihood, newbesttrees = likelihoods(besttrees,sequences)
#            print("@bestlike",newbestlikelihood)
#            newbestlikelihood, newbesttrees = likelihoods(besttrees,sequences, opt)
#            print("@bestlike",newbestlikelihood)
#            z = 0
#            for tr in bestindex:
#                Treelist[tr] = newbesttrees[z]
#                Likelihoods[tr] = newbestlikelihood[z]
#                z += 1
#            store_results(outputdir[it],'likelihood',Likelihoods)
#            store_results(outputdir[it],'treelist',Treelist)
#            store_results(outputdir[it],'starttrees',Treelist[:slen])
#            store_results(outputdir[it],'pathtrees',Treelist[slen:])


        tic = time.perf_counter()

        Pathtrees = masterpathtrees(StartTrees)
        store_results(outputdir[it],'Pathtrees',Pathtrees)
        print(f"\nlength of Pathtrees ---->  {len(Pathtrees)}")
        toc = time.perf_counter()
        timeing = toc - tic
        print(f"\nTime of generating pathtrees = {timeing}")
        Treelist= StartTrees+Pathtrees
        print(f"\nlength of treelist = StartTrees + Pathtrees = {len(StartTrees)} + {len(Pathtrees)} ---->  {len(Treelist)}")
        

        print(f'\n\n~~~~~~~~~~~~~~~~~~~~~~~~~ Likelihood ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
        tic = time.perf_counter()
        Likelihoods = likelihoods(Treelist,sequences)
        idx = plo.best_likelihoods(Likelihoods)
        toc = time.perf_counter()
        timeing = toc - tic
        print(f"\nTime of  Likelihood calculation = {timeing}")
        
        

        print(f'\n\n\n~~~~~~~~~~~~~~  Topologies of best {len(idx)}-trees  ~~~~~~~~~~~~~~\n\n')

        tic = time.perf_counter()
        BestTrees=[Treelist[i] for i in idx]
        store_results(outputdir[it],'BestTrees',BestTrees)

        dict_idx = dict(zip(range(0, len(idx)) , idx))
    
        print("\nusing unweighted Robinson-Foulds distance for plotting:")
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
        
        print(f"\n---> Number of Topologies = {len(Topologies)}")
        print(f"\n---> Topologies = {Topologies}\n")
        print(f"\n---> best_topo_like_idx = {best_topo_like_idx}")
        
        toc = time.perf_counter()
        timeing = toc - tic
        print(f"\n\nTime of  generating topologies = {timeing}")

        print(f'\n\n\n~~~~~~~~~~~~~~~~~~  Optimizing best trees  ~~~~~~~~~~~~~~~~~~\n\n')

        tic = time.perf_counter()

        optimized_BestTrees=[]
        for j,num in enumerate(best_topo_like_idx):
            tree_num = Treelist[num]
            mytree = tree.Tree()
            mytree.myread(tree_num,mytree.root)
            #            labels, sequences, variable_sites = ph.readData(datafile)        #Do we need this in each iteration???????
            mytree.insertSequence(mytree.root,labels,sequences)

            #~~~~~~~~~~~~~   paup optimize   OR  NelderMead optimize~~~~~~~~~~~~~
            if paup_optimize:
                optnewick = mytree.paupoptimize(datafile, filetype="RelPHYLIP")
            else:
                x,fval,iterations, tree1 = opt.minimize_neldermead(mytree, maxiter=None, maxiter_multiplier=200, initial_simplex=None, xatol=1e-4, fatol=1e-4, adaptive=False, bounds=[0,1e6])
                optnewick = tree1.treeprintvar().rstrip()   # assumes you have the new tree.py & rstrip() to remove \n from end of each new tree
#                print(f"\n\n~~~~~~~~~~~ {j}. tree[{num}] ~~~~~~~~~~~~\n")
#                print(f"Num of iterations = {iterations}\n\nOptimal Branchs&tips= {x}\n\nOptimal Likelihood= {fval}\n\nOptimized tree[{num}] = {optnewick}")
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            optimized_BestTrees.append(optnewick)
            
        toc = time.perf_counter()
        timeing = toc - tic
        print(f"\n\n\nTime of optimizing best trees = {timeing}")
        store_results(outputdir[it],'optimized_BestTrees',optimized_BestTrees)

        print(f'\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
        
        Likelihoods_optimized_BestTrees = likelihoods(optimized_BestTrees,sequences)

        sort_index_opt = sorted(range(len(Likelihoods_optimized_BestTrees)), key=lambda k: Likelihoods_optimized_BestTrees[k])
        Like_opt = [Likelihoods_optimized_BestTrees[i] for i in sort_index_opt]
        BestTrees_opt = [optimized_BestTrees[i] for i in sort_index_opt]
        Topologies_opt = [Topologies[i] for i in sort_index_opt]

        Treelist += BestTrees_opt
        print(f"length of treelist  ---->  {len(Treelist)}")
    
        Likelihoods += Like_opt
        
        
        if paup_tree:
            Treelist += paup
            print(f"length of treelist  after adding paup tree---->  {len(Treelist)}")
            Likelihood_PAUP = likelihoods(paup,sequences)
            Likelihoods += Likelihood_PAUP
            print(f"length of Likelihoods after adding paup tree ---->  {len(Likelihoods)}\n\n\n")
        elif paup_MAP:
            Treelist += paup_MAP
            print(f"length of treelist  after adding paup&MAP trees---->  {len(Treelist)}")
            Likelihood_PAUP_MAP = likelihoods(paup_MAP,sequences)
            Likelihoods += Likelihood_PAUP_MAP
            print(f"length of Likelihoods after adding paup&MAP trees ---->  {len(Likelihoods)}\n\n\n")



        store_results(outputdir[it],'optimized_BestTrees',BestTrees_opt)
        store_results(outputdir[it],'treelist',Treelist)
        store_results(outputdir[it],'Topologies_opt',Topologies_opt)
        store_results(outputdir[it],'likelihood',Likelihoods)
        


        #==========================  PLOT  ============================
        
#        if MDS_DEBUG:    #Not to see the plots, just the pathtrees of each iteration
        if plotfile != None:
            print(f'\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~ MDS plot~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
            tic3 = time.perf_counter()
            newtreelist = os.path.join(outputdir[it], 'treelist')
            if not keep:
                os.system(f'rm {GTPTERMINALLIST}')
                os.system(f'rm {GTPTREELIST}')
            if plotfile != None:
                if paup_tree:
                    bestlike = plo.best_likelihoods(Likelihoods[:-(len(BestTrees_opt)+1)])   # minus optimizaed trees and 1PAUP
                elif paup_MAP:
                    bestlike = plo.best_likelihoods(Likelihoods[:-(len(BestTrees_opt)+2)])   # minus optimizaed trees and 1PAUP&1MAP
                else:
                    bestlike = plo.best_likelihoods(Likelihoods[:-(len(BestTrees_opt))])   # minus optimizaed trees
                print(f"test ---->   len(bestlike) = {len(bestlike)}\n\n\n")
                n = len(Treelist)
                #~~~~~~~~~~  RF_distance   OR   GTP_distance~~~~~~~~~~~~~~~~~
                if RF_distances:
                    distances = wrf.RF_distances(n, newtreelist, type="weighted")
                    store_results(outputdir[it],'RF_distances',distances)
                else:
                    run_gtp(newtreelist, GTPTERMINALLIST)
                    os.system(f'mv pathtrees/gtp/output.txt {outputdir[it]}/')
                    distancefile = os.path.join(outputdir[it], 'output.txt')
                    distances = plo.read_GTP_distances(n,distancefile)
                    store_results(outputdir[it],'GTP_distances',distances)
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                if DEBUG:
#                    plo.plot_MDS(plotfile, N, n, distances, Likelihoods, bestlike, Treelist, Pathtrees)
#                plo.interpolate_grid(it, plotfile2[it], distances,Likelihoods, bestlike,Treelist, StartTrees,BestTrees_opt, Topologies_opt,NUMPATHTREES)
                plo.interpolate_rbf(it, plotfile2[it], distances,Likelihoods, bestlike,Treelist, StartTrees,BestTrees_opt, Topologies_opt,NUMPATHTREES)
    
        
        #==========================  Next Iteration  ============================
        if it1 < num_iterations-1:
            
            print(f'\n\n\n~~~~~~~~~~~~~~~~~~~  BoundaryTrees ~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
            tic = time.perf_counter()
            
            # make bifurcating and nonzero since we optimized trees that may be not biforcating now in the second iteration
            
            store_results(outputdir[it1],'BestTrees_opt',BestTrees_opt)     #Test
            
            T_bifurcate = bifurcating.bifurcating_newick(BestTrees_opt)
            store_results(outputdir[it1],'T_bifurcate',T_bifurcate)     #Test
            
            T_nonzero = nonzero_lengths(T_bifurcate)
            store_results(outputdir[it1],'T_nonzero',T_nonzero)     #Test
            
#            optimized_starttrees = nonzero_lengths(BestTrees_opt)
            StartTrees_forboundaries = T_nonzero
            Likelihoods_starttrees =  likelihoods(StartTrees_forboundaries, sequences)
            
            store_results(outputdir[it1],'StartTrees_forboundaries',StartTrees_forboundaries)
            distancefile = os.path.join(outputdir[it1], 'StartTrees_forboundaries')
            distances =  wrf.RF_distances(len(StartTrees_forboundaries), distancefile, type="weighted")
            Boundary_Trees = plo.boundary_convexhull(distances,Likelihoods_starttrees,StartTrees_forboundaries)
            
            store_results(outputdir[it1],'Boundary_Trees',Boundary_Trees)
            toc = time.perf_counter()
            timeing = toc - tic
            print(f"\n\nTime of generating BoundaryTrees = {timeing}")
            
            print(f'\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
            StartTrees = Boundary_Trees







#
#        if plotfile != None:
#            n = len(Treelist)
#            N = len(Pathtrees)
#            if not fast:
#                print("using GTP distance for plotting")
#                #distancefile = os.path.join(outputdir[it], 'output.txt')
#                distancefile = GTPOUTPUT
#                distances = plo.read_GTP_distances(n,distancefile)
#            else:
#                #print(newtreelist)
#                print("using weighted Robinson-Foulds distance for plotting")
#                distances = wrf.RF_distances(n, newtreelist)
#
#            if proptype:
#                idx = list(zip(*bestlike))[0]
#                X= plo.MDS(distances,2)
#                X1= X[idx, :]
#                #                print("\nlen of X =",np.shape(X1))
#                hull = ConvexHull(X1)
#                hull_indices = hull.vertices      # Get the indices of the hull points.
#                print("len of hull_indices =",len(hull_indices))
#                print("hull_indices =",hull_indices)
#                hull_idx = [idx[i] for i in hull_indices]
#                print("hull_idx  =",hull_idx )
#                hull_pts = X[hull_idx, :]       # These are the actual points.
#                Boundary_Trees = [Treelist[i] for i in hull_idx]
#                Boundary_Trees = [s.replace('\n', '') for s in Boundary_Trees]
#            else:
#                hull_idx = None
#
#            if DEBUG:
#                plo.plot_MDS(plotfile, N, n, distances, Likelihoods, bestlike, Treelist, Pathtrees)
#
#            plo.interpolate_grid(it, plotfile2[it], n, distances,Likelihoods, bestlike, Treelist, StartTrees, hull_idx)
#
#        if it1 < num_iterations:
#            StartTrees = [Treelist[tr] for tr in list(zip(*bestlike))[0]]
#            print("Number of start trees after an iteration: ",len(StartTrees))
