#!/usr/bin/env python
#
# optimization of branchlength using Nelder-Mead
# (c) Tara Khodaei
# MIT opensource license
#
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent
sys.path.append(str(file.parent))
import tree
import likelihood as like
import numpy as np

#xe=ye
#fxr=fr
#xbar=yc
#xr=yr
#rho=delta_r
#fxe=fe
#xc = y_oc
#xcc = y_ic

#Tara's likelihood hack


#def minimize_neldermead(func, x0, maxiter=None, initial_simplex=None,
#                        xatol=1e-4, fatol=1e-4, adaptive=False):
def minimize_neldermead(tree1, maxiter=None, initial_simplex=None, xatol=1e-4, fatol=1e-4, adaptive=False): #x0:edges lengths of the tree
    delegates = []
    tree1.root.name='root'
    tree1.delegate_extract(tree1.root,delegates)
    #print(tree1.root.name)
    #print(delegates)
    #sys.exit()
    x0,s0,clean0,type0 = tree.extract_delegate_branchlengths(delegates)
    #x0 = tiplen+edgelen  # list of all branch lengths, best would be if these map directly into the tree
    #l = len(tiplen)
    #print(x0)
    x0 = np.asfarray(x0).flatten() # is this needed? because the x0 should be flat
    #print(x0)
    if adaptive:
        dim = float(len(x0))
        delta_r = 1
        chi = 1. + 2./dim
        psi = 0.75 - 1./(2*dim)
        sigma = 1. - 1./dim
    else:
        delta_r = 1
        chi = 2
        psi = 0.5
        sigma = 0.5
    
    nonzdelt = 0.05
    zdelt = 0.00025
    
    
    
    if initial_simplex is None:
        N = len(x0)
        sim = np.empty((N + 1, N), dtype=x0.dtype)
        sim[0] = x0
        for k in range(N):
            y = x0.copy() #np.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + nonzdelt)*y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        sim = np.asfarray(initial_simplex).copy()
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError(f"`initial_simplex` should be an array of shape (N+1,N) [N={N}]")
        if len(x0) != sim.shape[1]:
            raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
        N = sim.shape[1]
    
    
    
    # If neither are set, then set both to default
    if maxiter is None :
#        maxiter = N * 200
        maxiter = N * 20       #New

    # initialization of f_sim vector
    f_sim = np.empty((N + 1,), float)
    temp = -tree1.delegate_calclike(delegates)
    for k in range(N + 1):
        #tree1 = splittree.print_newick_string(tips,edges,sim[k][:l],sim[k][l:])    #new2
        #f_sim[k] = likelihoods(tree1,sequences,labels)
        f_sim[k] = temp

    ind = np.argsort(f_sim)
    f_sim = np.take(f_sim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim, ind, 0)
#    print(f"\n\nTest---> len(f_sim) = {len(f_sim)}")
    #print(f"\n\nTest---> f_sim = {f_sim}")

    
    iterations = 1
    
    while (iterations < maxiter):
        if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and
            np.max(np.abs(f_sim[0] - f_sim[1:])) <= fatol):
            break
            
        y_c = np.add.reduce(sim[:-1], 0) / N       #centroid
        y_r = y_c + delta_r *( y_c - sim[-1])        #reflect

        #tree1 = splittree.print_newick_string(tips,edges,y_r[:l],y_r[l:])    #new2
        #f_r = likelihoods(tree1,sequences,labels)
        tree.instruct_delegate_branchlengths(y_r,delegates)
        f_r = -tree1.delegate_calclike(delegates)
        doshrink = 0
        
        if f_r < f_sim[0]:    #Expand
            y_e = y_c + (delta_r * chi)*(y_c- sim[-1])
            
            #tree1 = splittree.print_newick_string(tips,edges,y_e[:l],y_e[l:])    #new2
            #f_e = likelihoods(tree1,sequences,labels)
            tree.instruct_delegate_branchlengths(y_e,delegates)
            f_e = -tree1.delegate_calclike(delegates)
#            print(f"\n\nTest---> f_r = {f_r}")

            if f_e < f_r:
                sim[-1] = y_e
                f_sim[-1] = f_e
            else:
                sim[-1] = y_r
                f_sim[-1] = f_r
        else:  # fsim[0] <= fr
            if f_r < f_sim[-2]:
                sim[-1] = y_r
                f_sim[-1] = f_r
            else:  # fr >= fsim[-2]    #contract
                if f_r < f_sim[-1]:    #outside contraction
                    y_oc = y_c + (psi * delta_r) *( y_c - sim[-1])
                    #tree1 = splittree.print_newick_string(tips,edges,y_oc[:l],y_oc[l:])    #new2
                    #f_oc = likelihoods(tree1,sequences,labels)
                    tree.instruct_delegate_branchlengths(y_oc,delegates)
                    f_oc = -tree1.delegate_calclike(delegates)
#                    print(f"\n\nTest---> f_oc = {f_oc}")
                    if f_oc <= f_r:
                        sim[-1] = y_oc
                        f_sim[-1] = f_oc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    y_ic = y_c + psi *(-y_c + sim[-1])     #inside contraction
                    #tree1 = splittree.print_newick_string(tips,edges,y_ic[:l],y_ic[l:])    #new2
                    #f_ic = likelihoods(tree1,sequences,labels)
                    tree.instruct_delegate_branchlengths(y_ic,delegates)
                    f_ic = -tree1.delegate_calclike(delegates)
                    if f_ic < f_sim[-1]:
                        sim[-1] = y_ic
                        f_sim[-1] = f_ic
                    else:
                        doshrink = 1
            
                if doshrink:      #shrink
                    for j in range(1, N + 1):
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        #tree1 = splittree.print_newick_string(tips,edges,sim[j][:l],sim[j][l:])    #new2
                        #f_sim[j] = likelihoods(tree1,sequences,labels)
                        tree.instruct_delegate_branchlengths(sim[j],delegates)
                        f_sim[j] = -tree1.delegate_calclike(delegates)

        ind = np.argsort(f_sim)
        sim = np.take(sim, ind, 0)
        f_sim = np.take(f_sim, ind, 0)
        #print(f"\n\niteration #{iterations}\nsim[0] = {sim[0]}\n\n")
        iterations += 1
    
    
    x = sim[0]
    fval = np.min(f_sim)
    
    return (x,fval,iterations, tree1)


# this does not work with the changes made to the minimize_neldermead function
#if __name__ == "__main__":
#    def f(x):   # The rosenbrock function
#        return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
#
#    x0= [2, -1]
#
#    x,fval,iterations = minimize_neldermead(f, x0, maxiter=None, initial_simplex=None,
#                        xatol=1e-4, fatol=1e-4, adaptive=False)
#    print(f"x_ optimal = {x}\n\nf_ optimal = {fval}\n\nnum of iterations = {iterations}")

