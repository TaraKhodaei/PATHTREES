# Calculate likelihood on trees
# IMPORTANT: all arrays and matrices need to be numpy arrays
#
# -JukesCantor       : defines mutation Jukes Cantor rate matrix  
#                      and basefrequencies 
# -assignCond        : (internal) assigns conditionals based on nucleotide
# -tipCondLikelihood : sets all conditionals for tips using sequence string
# -logLikelihood     : calculates log likelihood at root using 
#                      basefrequencies and root conditionals
# -condLikelihood    : calculates the conditionals for one branch 
#                      (the intermediate values)
# -nodeLikelihood    : caculates the conditionals for a node given 
#                      left and right nodes
# -test              : when this file is called alone then it runs this test 
import sys
import numpy as np    # this loads numpy as shortcut np
import scipy as sc    #
import scipy.linalg   #

#
# set this to true if you want intermediate output
DEBUG = False

def JukesCantor():
	Q =  np.array(
	[
	[-1.0   ,1.0/3.0,1.0/3.0,1.0/3.0],
	[1.0/3.0,   -1.0,1.0/3.0,1.0/3.0],
	[1.0/3.0,1.0/3.0,   -1.0,1.0/3.0],
	[1.0/3.0,1.0/3.0,1.0/3.0,   -1.0]
	])
	basefreq = np.array([0.25,0.25,0.25,0.25])
	return Q,basefreq



def assignCond(site):
	if site.upper()=='A':
		return [1.0,0.0,0.0,0.0]
	if site.upper()=='C':
		return [0.0,1.0,0.0,0.0]
	if site.upper()=='G':
		return [0.0,0.0,1.0,0.0]
	if site.upper()=='T':
		return [0.0,0.0,0.0,1.0]
	return [1.0,1.0,1.0,1.0]

def tipCondLikelihood(sequence):
	seq =  list(sequence)
	cond = [assignCond(s) for s in seq]
	return np.array(cond)
		
def logLikelihood(g,basefreq):
    gT = g.T
    g0 = np.dot(basefreq,gT)
    if DEBUG:
        print("g0>0", g0, basefreq, g)
        for gi in g0:
           if gi < 0:
               sys.exit()
    
    if g0.any() <=0:
        print(f"g0={g0} basefreq={basefreq} gT={gT}")
        return -Inf
    try:
        #print("g0=",g0)
        ssum = np.sum(np.log(g0))
    except:
        #print("*g0=",g0)
        #sys.exit()
        pass
    return ssum



# conditional likelihood takes g and calculates h
# g can be a 2D-vector of conditional likelihoods
#
def condLikelihood(g, Q, t):
        #calculate probability transition matrix p(t) = exp(Q t)
        #print(Q)
        #print(t)
        try:
                p = scipy.linalg.expm(Q * t)
        except:
                print(f'Q={Q} t={t} g={g}')
                if np.isnan(t):
                        t=0.0
                        p = scipy.linalg.expm(Q * t)
                        return np.dot(g,p)
                else:
                        sys.exit()
        if DEBUG:
                print ("Transition probability p with branchlength", t)
                #print (p)
                #print ('g-----------------')
                print (g)
                #print ('-----------------')
                #print (t)
                print ('-----------------')
        return np.dot(g,p)

def nodeLikelihood(ga,gb,ta,tb,Q):
	ha = condLikelihood(ga, Q, ta)
	hb = condLikelihood(gb, Q, tb)
	gd = ha * hb # elementwise multiplication
	if DEBUG:
		print ("ha,hb=",ha,hb)
		print ("gd=",gd)
	return gd

def test():
	basefreq = np.array([0.25,0.25,0.25,0.25]) # assuming JC model
						   # tip specifications
	ga = tipCondLikelihood("ACGT")
	gb = tipCondLikelihood("AAGT")
	gc = tipCondLikelihood("CGGT")
	# rate matrix for Jukes Cantor
	Q = np.array(
	[
	[-1.0   ,1.0/3.0,1.0/3.0,1.0/3.0],
	[1.0/3.0,   -1.0,1.0/3.0,1.0/3.0],
	[1.0/3.0,1.0/3.0,   -1.0,1.0/3.0],
	[1.0/3.0,1.0/3.0,1.0/3.0,   -1.0]
	])

	tad = 0.3
	tbd = 0.3
	tde = 0.3
	tce= 0.6
	gd = nodeLikelihood(ga,gb,tad,tbd,Q)
	ge = nodeLikelihood(gd,gc,tde,tce,Q)
	l = logLikelihood(ge,basefreq)
	if DEBUG:
                print ("log Likelihood = ", l)


if __name__ == '__main__':
	print ("test run")
	test()







