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

import numpy as np
import time
#import shutil

GTPTREELIST = 'gtptreelist' # a pair of trees formed from the master treelist
GTPTERMINALLIST = 'terminal_output_gtp'  #GTP terminal output
GTPOUTPUT = 'output.txt' #GTP output file , check later in the source!
NUMPATHTREES = 10  #number of trees in path
GTP = os.path.join(parent, 'pathtrees','gtp')

def create_treepair(ti,tj,pairtreelist):
    f = open(pairtreelist,'w')
    f.write(ti.strip())
    f.write("\n")
    f.write(tj.strip())
    f.write('\n')
    f.close()

def run_gtp(gtptreelist,gtpterminallist,gtpoutput):
    os.system(f"cd  {GTP}; java -jar gtp.jar -v -o {gtpoutput} {gtptreelist} > {gtpterminallist}")
    
def masterpathtrees(treelist): #this is the master treelist
    # loop over treelist:
    allpathtrees = []
    print(f"masterpathtrees: {len(treelist)} trees")
    print(f"gtptreelist={GTPTREELIST}")
    print(f"gtpterminallist={GTPTERMINALLIST}")
    for i,ti in enumerate(treelist):
        for j,tj in enumerate(treelist):
            if j<=i:
                continue
            #form a treelist of the pair
            create_treepair(ti,tj,GTPTREELIST) #writes to a file GTPTREELIST
            #print("run gtp")
            run_gtp(GTPTREELIST, GTPTERMINALLIST, GTPOUTPUT)
            #print(GTPTREELIST, GTPTERMINALLIST, GTPOUTPUT)
            mypathtrees = pt.internalpathtrees(GTPTREELIST, GTPTERMINALLIST, NUMPATHTREES)
            #print("LOOP",i,j)
            allpathtrees.extend(mypathtrees)
            #save or manipulate your pathtrees
    return [a.strip() for a in allpathtrees]

def likelihoods(trees,sequences, opt=False):
    #global labels
    if DEBUG:
        print("Likelihood:",f"number of trees:{len(trees)}")
        print("Likelihood:",f"number of sequences:{len(sequences)}")
        print("Likelihood:",f"optimize:{opt}")
    likelihood_values=[]
    newtrees = [] # for opt=True
    lt = len(trees)
    for i,newtree in enumerate(trees):
        t = tree.Tree()
        t.myread(newtree,t.root)
        t.insertSequence(t.root,labels,sequences)
        
        #setup mutation model
        # the default for tree is JukesCantor,
        # so these two steps are not really necessary
        Q, basefreqs = like.JukesCantor()
        t.setParameters(Q,basefreqs)
        #calculate likelihood and return it
        if opt:  
            if NR:
                t.optimizeNR()
            else:
                #t.optimize()
                pnewick = t.paupoptimize(datafile,filetype)
                pnewtree = Tree()
                pnewtree.root.name='root'
                pnewtree.root.blength=0.0
                pnewtree.myread(pnewick,newtree.root)
                pnewtree.insertSequence(pnewtree.root,labels,sequences)
                pnewtree.likelihood()
                t = pnewtree
            if DEBUG:
                print(f"optimized tree {i} of {lt} with lnL={t.lnL}")
            likelihood_values.append(t.lnL)
            with split.Redirectedstdout() as newick:
                t.myprint(t.root,file=sys.stdout)    
            newtrees.append(str(newick)+';')
        else:
            t.likelihood()
            likelihood_values.append(t.lnL)
            if DEBUG:
                print("Likelihood:",f"lnL={t.lnL} {newtree[:50]}")
    return likelihood_values, newtrees

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
    DEBUG = False
    args = myparser() # parses the commandline arguments
    start_trees = args.STARTTREES
    datafile = args.DATAFILE
    outputdir = args.outputdir
    keep = args.verbose == True
    num_random_trees = args.num_random_trees
    outgroup = args.outgroup
    num_iterations = args.num_iterations+1
    plotfile = args.plotfile
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
    NUMBESTTREES = args.NUMBESTTREES
    STEPSIZE = 1  # perhaps this should be part of options

    #    print(args.plotfile)
    print(args)
    print(datafile)
    print(ttype)
    labels, sequences, variable_sites = ph.readData(datafile, ttype)
    #print(labels)
    #print(variable_sites)
    #sys.exit()
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
    print(f'Calculating paths through tree space')
    tictotal = time.perf_counter()
    with open(start_trees,'r') as f:
        StartTrees = [line.strip() for line in f]
    # iterations, this is done at least once for standard operation
    #
    for it1 in range(1,num_iterations):
        tic = time.perf_counter()
        it = it1-1
        print(f'Iteration {it1}')
        #store_results(outputdir[it],f'starttrees-{it1}',StartTrees)
        GTPTREELIST = os.path.join(current,outputdir[it],'gtptreelist') # a pair of trees formed from the master treelist
        GTPTERMINALLIST = os.path.join(current,outputdir[it],'terminal_output_gtp')  #GTP terminal output
        GTPOUTPUT = os.path.join(current,outputdir[it],'output.txt')  #GTP output file
        Pathtrees = masterpathtrees(StartTrees)
        slen = len(StartTrees)
        Treelist= StartTrees+Pathtrees
        Likelihoods, newtreelist  = likelihoods(Treelist,sequences,allopt)
        if allopt:
            StartTrees = newtreelist[:slen]
            Pathtrees = newtreelist[slen:]
            Treelist = newtreelist[:]
        Likelihoods = [like if like != -np.inf else -10**8 for like in Likelihoods]
        store_results(outputdir[it],'likelihood',Likelihoods)
        store_results(outputdir[it],'treelist',Treelist)
        store_results(outputdir[it],'starttrees',StartTrees)
        store_results(outputdir[it],'pathtrees',Pathtrees)
        toc = time.perf_counter()
        time1 = toc - tic
        print(f"Time of generating pathtrees results = {time1}")
        tic2 = time.perf_counter()

        newtreelist = os.path.join(outputdir[it], 'treelist')
        if not fast:
            print('Calculate geodesic distance among all pathtrees')
            run_gtp(newtreelist, GTPTERMINALLIST,GTPOUTPUT)
            #os.system(f'mv pathtrees/gtp/output.txt {outputdir[it]}/')
            #if not keep:
            #    os.system(f'rm {GTPTERMINALLIST}')
            #    os.system(f'rm {GTPTREELIST}')
            toc2 = time.perf_counter()
            time2 = toc2 - tic2
            print(f"Time of GTP distances of all trees = {time2}")
        
        bestlike = plo.bestNstep_likelihoods(Likelihoods,NUMBESTTREES,STEPSIZE)

        if opt:
            bestindex = list(zip(*bestlike))[0]
            besttrees = np.take(Treelist,bestindex)
            newbestlikelihood, newbesttrees = likelihoods(besttrees,sequences)
            print("@bestlike",newbestlikelihood)
            newbestlikelihood, newbesttrees = likelihoods(besttrees,sequences, opt)
            print("@bestlike",newbestlikelihood)
            z = 0
            for tr in bestindex:
                Treelist[tr] = newbesttrees[z]
                Likelihoods[tr] = newbestlikelihood[z]
                z += 1
            store_results(outputdir[it],'likelihood',Likelihoods)
            store_results(outputdir[it],'treelist',Treelist)
            store_results(outputdir[it],'starttrees',Treelist[:slen])
            store_results(outputdir[it],'pathtrees',Treelist[slen:])

                
        if plotfile != None:
            n = len(Treelist)
            N = len(Pathtrees)
            if not fast:
                print("using GTP distance for plotting")
                #distancefile = os.path.join(outputdir[it], 'output.txt')
                distancefile = GTPOUTPUT
                distances = plo.read_GTP_distances(n,distancefile)
            else:
                #print(newtreelist)
                print("using weighted Robinson-Foulds distance for plotting")
                distances = wrf.RF_distances(n, newtreelist)

            if proptype:
                idx = list(zip(*bestlike))[0]   
                X= plo.MDS(distances,2)
                X1= X[idx, :]
                #                print("\nlen of X =",np.shape(X1))
                hull = ConvexHull(X1)
                hull_indices = hull.vertices      # Get the indices of the hull points.
                print("len of hull_indices =",len(hull_indices))
                print("hull_indices =",hull_indices)
                hull_idx = [idx[i] for i in hull_indices]
                print("hull_idx  =",hull_idx )
                hull_pts = X[hull_idx, :]       # These are the actual points.
                Boundary_Trees = [Treelist[i] for i in hull_idx]
                Boundary_Trees = [s.replace('\n', '') for s in Boundary_Trees]
            else:
                hull_idx = None

            if DEBUG:
                plo.plot_MDS(plotfile, N, n, distances, Likelihoods, bestlike, Treelist, Pathtrees)
                
            plo.interpolate_grid(it, plotfile2[it], n, distances,Likelihoods, bestlike, Treelist, StartTrees, hull_idx)

        if it1 < num_iterations:
            StartTrees = [Treelist[tr] for tr in list(zip(*bestlike))[0]]
            print("Number of start trees after an iteration: ",len(StartTrees))
