#!/usr/bin/env python3

DEBUG=False
# node and tree class
# class Node defines:
#     -Node (see __init__)
#     -tip          : defines tip node and sets label
#     -interior     : defines interior node and sets connections left and right
#     -myprint      : prints name and branchlength associated with a node
#     -debugprint   : prints the content of a node
# class Tree defines:
#     -Tree (see __init__) : defines root, basefrequencies and JukesCantor rate matrix
#     -myprint      : prints the tree in NEWICK format
#     -myread       : reads a NEWICK string and creates a tree
#     -printTiplabels: prints tip labels
#     -insertSequence: uses a list of labels and sequences to match the
#                      labels with the tip labels and insert the conditioinal
#                      likelihoods into the tips according to the data
#                      in the sequences list
#     -condLikelihood: internal function to calculate the conditionals in all nodes
#                      (downpass)
#     -likelihood    : calculates the log likelihood of the whole tree
#                      (uses Tree.condLikelihood)
#     -optimizeNR    : optimize branch length (likelihood) using Newton-Raphson {works on some trees}
#     -optimize      : use Nelder Mead optimization
#     -setParameters : sets the mutation rate matrix and basefrequencies
#     -printLnL      : prints log likelihood of tree
#     -getLnL        : returns log likelihood
# utility functions:
# - getName          : extracts the lable from the newickstring
# - istip            : returns true if node is a tip
# PB Oct 2011



import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent
sys.path.append(str(file.parent))
#try:
#    sys.path.remove(str(parent))
#except ValueError: # Already removed
#    pass
#print(sys.path)


import numpy as np
from numpy.random import default_rng
from scipy.optimize import linprog
from scipy.optimize import minimize
import likelihood as like
import optimize as opt

#=========================================  Peter code  ==========================================
#import re


def getName(s):
    name = ""
    #s = re.sub('[&.*?]', '', s)
    for j in range(len(s)):
        if((s[j]==')') | (s[j]=='(') | (s[j]==':') | (s[j]==' ') | (s[j]==',')):
            return j,name
        else:
            name= name + s[j]
    return j,name



def istip(p):
    #print("istip:",p)
    if((p.left == -1) & (p.right == -1) & (p.name != "root")):
        return True
    else:
        return False



class Node:
    """
    Node class: this is a container for content that is saved
    at nodes in a tree
    """
    def __init__(self):
        """
        basic node init
        """
        self.name = ""
        self.left = -1
        self.right = -1
        self.ancestor = -1
        self.blength = -1
        self.sequence = []
        self.clean = False
        
    def tip(self,name):
        """
        sets the name of a tip
        """
        self.name = name
        self.left = -1
        self.right = -1
        self.ancestor = -1
        self.blength = -1
    
    def interior(self,left,right):
        """
        connects an interior node with the two descendents
        """
        self.name = ""
        self.left = left
        self.right = right
        self.ancestor = -1
        self.blength = -1

    def optimize_branch(self,thetree):
        f = thetree.likelihood()
        store = self.blength
        #print("before change:",f)
        if store < 0.000001:
            h = 0.000001
        else:
            h = 0.25 * store
        self.blength = store - h
        if self.blength < 0.0:
            self.blength = 0.0
        f0 = thetree.likelihood()
        #print("f0:",f0, "[blen=", store - h,"]")
        self.blength = store + h
        f2 = thetree.likelihood()
        if DEBUG:
            print(f'f2={f2} f0={f0} h={h} [f={f}]')
        if f2 == -np.inf and f0 == -np.inf:
            return
        if np.abs(f2-f0) > 0.000001 and h > 0.000001:
            fdash = (f2 - f0)/h
            ftwodash = ((f2 - 2*f + f0)/(h*h))
            self.blength = store - fdash / ftwodash
            if self.blength < 0.0:
                self.blength = 0.0
        else:
            fdash = 0.0
            ftwodash=1.0
            self.blength = store
        x = thetree.likelihood()
        #print(f"final:b={self.blength},borig={store}, f'={fdash}/f''={ftwodash}, [{f}, {f0}, {f2}, lnL={x}")
        mult = 0.5
        counter = 0
        while x == -np.inf or x < f: # or x < f0 or x < f2:
            self.blength = store - mult * fdash / ftwodash
            fdash /= 2.0
            mult /= 2.0
            if self.blength < 0.0:
                self.blength = 0.0
            x = thetree.likelihood()
            if fdash < 0.00001 or counter > 20:
                break
            counter += 1
            #if DEBUG:
            #print(f"@@@final:b={self.blength},borig={store}, f'={fdash}/f''={ftwodash}, [{f}, {f0}, {f2}, lnL={x}")
        #else:
        #    print(f"opt lnL {x}")

    def myprint(self,file=sys.stdout):
        """
        Prints the content of a node: name if any and branchlengths if any
        """
        if(self.name!=""):
            print (self.name,end='',file=file)
        if(self.blength != -1):
            print (":%s" % str(self.blength),end='',file=file)

    def debugprint(self):
        """
        Prints the content of a node: name if any and branchlengths if any
        """
        print (self.name,end=' ')
        print (self.left,end=' ')
        print (self.right,end=' ')
        print (self.ancestor,end=' ')
        print (":%s" % str(self.blength),end=' ')
        print (self.sequence)

class Tree(Node):
    """
    Node class: this is a container for content that is saved
    at nodes in a tree
    """
    i = 0
    
    def __init__(self):
        self.root = Node()
        self.root.name = "root"
        self.Q, self.basefreqs = like.JukesCantor()
    
    
    def myprint(self,p, file=sys.stdout):
        """
        prints a tree in Newick format
        """
        if(p.left != -1):
            print ("(",end='',file=file)
            self.myprint(p.left,file=file)
            print (",",end='',file=file)
        if(p.right != -1):
            self.myprint(p.right,file=file)
            print (")",end='',file=file)
        p.myprint(file=file)
        print ("",end='',file=file)
    
    def remove_internal_labels(self,p):
        """
            prints a tree in Newick format
            """
        if p.left != -1 and p.right != -1:
            p.name = ""
        if(p.left != -1):
            self.remove_internal_labels(p.left)
        if(p.right != -1):
            self.remove_internal_labels(p.right)


    def myread(self,newick, p):
        """
        reads a tree in newick format
        """
#        print("@",self.i)
#        print(newick[self.i:])
        #if newick[self.i] =='[':
        #    while newick[self.i]!=']':
        #        self.i += 1
        #    self.i += 1

        if(newick[self.i]=="("):
            q = Node()
            p.left = q
            self.i += 1
            self.myread(newick,q)
            if(newick[self.i]!=','):
                print ("error reading, failed to find ',' in %s" % newick)
            self.i += 1
         #   if newick[self.i] =='[':
         #       while newick[self.i]!=']':
         #           self.i += 1
         #       self.i += 1
            q = Node()
            p.right = q
            q.ancestor = p
            self.myread(newick,q)
            #if p is ’root’ of unrooted tree then
            if(newick[self.i]!=')'):
                if(newick[self.i]==','): # unrooted tree?
                    print("unrooted")
                    q = Node()
#                    print(p.right)
#                    print(p.left)
#                    print(p.ancestor)
                    p.myprint()
#                    print("####end")
                    p.ancestor = q
                    q.right = p
                    p = Node()
                    q.left = p
                    self.root = q
                    self.i += 1
                    self.myread(newick,p)
#                    print("***",end=' ')
                    p.myprint()
                    q.myprint()
                else:
                    print ("error reading, failed to find ')' in %s" % newick)
            self.i += 1
          #  if newick[self.i] =='[':
          #      while newick[self.i]!=']':
          #          self.i += 1
          #      self.i += 1

        if newick[self.i] == ';':
#            print("reached end")
            return
        
        j, p.name = getName(newick[self.i:])
#        print ("@@@", self.i, p.name, newick[self.i:])
        self.i = self.i + j
        if(newick[self.i] == ":"):
            self.i = self.i+1
            j, xx = getName(newick[self.i:])
#            print(float(xx),float(xx.strip("; \n\t")))
            p.blength = float(xx.strip("; \n\t"))
            self.i = self.i + j

    def printTiplabels(self,p):
        if not(istip(p)):
            self.printTiplabels(p.left)
            self.printTiplabels(p.right)
        else:
            print (p.name)

    def insertSequence(self, p, label, sequences):
        #print("insert:", label)
        if not(istip(p)):
            self.insertSequence(p.left,label,sequences)
            self.insertSequence(p.right,label,sequences)
        else:
            the_name = p.name.strip()
            the_name.replace(' ','_')
            if DEBUG:
                print("@@@ tipname", the_name, "Labels:",label)
            pos = label.index(the_name.strip())
            p.sequence = like.tipCondLikelihood(sequences[pos])


    def condLikelihood(self,p):
        if not(istip(p)):
            #print ("\nbefore left: p=",p,": p.left=",p.left,"p.left.name=","'"+p.left.name+"'")
            self.condLikelihood(p.left)
            #print ("\nbefore right: p=",p,": p.right=",p.right, "p.right.name=","'"+p.right.name+"'")
            self.condLikelihood(p.right)
            p.sequence = like.nodeLikelihood(p.left.sequence,
                                             p.right.sequence,
                                             p.left.blength,
                                             p.right.blength,
                                             self.Q)
        #print ("\ninternal node: p=",p,"\n      condlike[first site]=",p.sequence.tolist()[:1])

    


        
    def likelihood(self):
        self.condLikelihood(self.root)
        self.lnL = like.logLikelihood(self.root.sequence,self.basefreqs)
        return self.lnL

    def delegate_extract(self,p,delegate):
        if not(istip(p)):
            self.delegate_extract(p.left,delegate)
            self.delegate_extract(p.right,delegate)
            type=''
            if istip(p.left):
                p.left.clean=True
                type+='t'
            else:
                p.left.clean=False
                type+='i'
            if istip(p.right):
                p.right.clean=True
                type+='t'
            else:
                p.right.clean=False
                type+='i'
            p.clean = False
            if p.name=='root':
                type+='r'
            else:
                type+='i'
            delegate.append([p.left,p.right,p, type])

    def delegate_calclike(self,delegates):
        blen=[]
        seq =[]
        done = False
        while not done:
            for d in delegates:
                #print(d)
                if d[0].clean and d[1].clean:
                    d[2].sequence = like.nodeLikelihood(d[0].sequence,d[1].sequence,d[0].blength, d[1].blength, self.Q)
                    d[2].clean = True
                    if d[3][2]=='r':
                        done = True
                        break
                else:
                    continue
        #print('before lnL')        
        self.lnL = like.logLikelihood(self.root.sequence,self.basefreqs)
        #print(self.lnL)
        return self.lnL
        
    def optimizeNR(self):
        self.optimize_branch(self.root)
        #print("finished1",self.likelihood())
        #self.optimize_branch(self.root)
        #print("finished2",self.likelihood())
        return self.lnL
        
    def optimize_branch(self,p):
        if not(istip(p)):
            self.optimize_branch(p.left)
            p.optimize_branch(self)   
            self.optimize_branch(p.right)
            p.optimize_branch(self)
        else:
            p.optimize_branch(self)   
            
    def scioptimize(self):
        # minimize using Nelder-Mead
        delegates=[]
        self.delegate_extract(self.root,delegates)
        x0,s0,clean0,type0 = extract_delegate_branchlengths(delegates)
        bounds = [(0,100)]*len(x0)
        result = minimize(neldermeadlike, x0, args=(delegates,self),method='Nelder-Mead',bounds=bounds)
        #result = minimize(neldermeadlike, x0, args=(delegates,self),method='L-BFGS-B',bounds=bounds)
        x = result['x']
        iterations = result['nit']
        print(x,iterations)
        #x,fval,iterations, tree1 = minimize(neldermeadlike, x0, args=(delegates,self),method='Nelder-Mead')
        #,maxiter=None, initial_simplex=None, xatol=1e-2, fatol=1e-2, adaptive=False)
        instruct_delegate_branchlengths(x,delegates)
        self.delegate_calclike(delegates)
        #if DEBUG:
        print("scioptimize: optimize tree likelihood:", self.lnL,iterations)
        return(self)

    def optimize(self):
        # minimize using Nelder-Mead
        x,fval,iterations, tree1 = opt.minimize_neldermead(self, maxiter=None, initial_simplex=None, xatol=1e-2, fatol=1e-2, adaptive=False)
        delegates =[]
        tree1.delegate_extract(tree1.root,delegates)
        instruct_delegate_branchlengths(x,delegates)
        tree1.lnL = -fval
        #if DEBUG:
        print("optimize tree likelihood:", -fval,iterations)
        return(tree1)

    
    def setParameters(self,Q,basefreqs):
        self.Q = Q
        self.basefreqs = basefreqs
    
    def printLnL(self):
        print (self.lnL)
    
    def getLnL(self):
        return self.lnL
    

    #=========================================  Tara code  ==========================================

    def getTips(self,p, tips, tipslength):
        if not(istip(p)):
            tips, tipslength = self.getTips(p.left, tips, tipslength)
            tips, tipslength = self.getTips(p.right, tips, tipslength)
        else:
            tips.append(p)
            tipslength.append(p.blength)
        return(tips, tipslength)



    def get_edges(self,p,edges,edgelengths):
        tips=[]
        tipslength=[]
        if not(istip(p)):
            tips, tipslength = self.getTips(p, tips, tipslength)
            names=[tips[i].name for i in range(len(tips))]
            edges.append(names)
            edgelengths.append(p.blength)
            self.get_edges(p.right,edges,edgelengths)
            self.get_edges(p.left,edges,edgelengths)
        return(edges[1:], edgelengths[1:])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Kendall ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def findPath(self, root, path, n):
        if root is None:
            return False
        path.append(root)
        if root.name == n :
            return True
        if ((root.left != -1 and self.findPath( root.left, path, n)) or
                (root.right!= -1 and self.findPath( root.right, path, n))):
            return True
        path.pop()
        return False



    def Root_MRCA(self, root, n1, n2):
        path1 = []
        path2 = []
        if (not self.findPath(root, path1, n1.name) or not self.findPath(root, path2, n2.name)):
            return -1
        i = 0
        sum = np.zeros(len(path1))
        while(i < len(path1) and i < len(path2)):
            if path1[i] != path2[i]:
                break
            i += 1
            sum[i]= sum[i-1]+(path1[i]).blength
        return(i-1, sum[i-1])



    def PairTips(self,p, tips):   # *** New code changes here ***
        pairlist=[]
        for i in range(len(tips)):
            for j in range(i+1, len(tips)):
                pairlist.append([tips[i], tips[j]])
        pairlist.sort(key=lambda x: (x[0].name, x[1].name))
        return pairlist
    


    def MetricsVectors(self, newick):    #*** New code changes here ***
        m=[]
        M=[]
        tips=[]
        tipslength=[]
        p = self.root
        mytree.myread(newick,p)
        tips, tipslength = mytree.getTips(p, tips, tipslength)
        tipsandlength = list(zip(tips, tipslength))
        tipsandlength.sort(key=lambda x: x[0].name)
        tips, tipslength = list(zip(*tipsandlength))
        pairlist = mytree.PairTips(p, tips)
        for pair in pairlist:
            numEdgePath, LengthPath = mytree.Root_MRCA(p, pair[0], pair[1])
            m.append(numEdgePath)
            M.append(LengthPath)
        ptips=[1] * len(tips)
        m=m+ptips
        M=M+list(tipslength)
        return (m, M)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Outside class functions  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def neldermeadlike(x0,*args): # args is (delegates,tree1)
    
    delegates,tree1 = args
    
    instruct_delegate_branchlengths(x0,delegates)
    return -tree1.delegate_calclike(delegates)

def extract_delegate_branchlengths(delegates):
    x0 = []
    s0=[]
    clean0=[]
    type0=[]
    for d in delegates:
        x0.extend([d[0].blength,d[1].blength])
        s0.append([d[0].sequence,d[1].sequence,d[2].sequence])
        clean0.append([d[1].clean,d[2].clean])
        type0.append(d[3])
    return x0,s0,clean0,type0

def instruct_delegate_branchlengths(x0, delegates):
    z=0
    for d in delegates:
        if d[0].blength != x0[z]:
            d[0].blength = x0[z]
            if d[3][0]!='t':
                d[0].clean = False
            d[2].clean = False
        z += 1
        if d[1].blength != x0[z]:
            d[1].blength = x0[z]
            if d[3][1]!='t':
                d[1].clean = False
            d[2].clean = False
        z += 1

def TreesDistance(m, M, c):      # 0<=c <=1     (Peter: For scaling)
    s0=np.sum(np.array(m[0]))
    S0=np.sum(np.array(M[0]))
    s1=np.sum(np.array(m[1]))
    S1=np.sum(np.array(M[1]))
    v1=(1-c)*np.array(m[0])/s0+c*np.array(M[0])/S0
    v2=(1-c)*np.array(m[1])/s1+c*np.array(M[1])/S1
    v11=(1-c)*np.array(m[0])+c*np.array(M[0])
    v22=(1-c)*np.array(m[1])+c*np.array(M[1])
    distance=np.linalg.norm(np.asarray(v1)-np.asarray(v2), ord=2)   #Scaled
#    distance=np.linalg.norm(np.asarray(v11)-np.asarray(v22), ord=2)  #Unscaled
    return distance


def skip_comment(tree):      # to remove the brackets []
    i=0
    tree_new=''
    while i <len(tree):
        if tree[i]=="[":
            while tree[i] != "]":
                i += 1
                tree_new = tree_new +''
        else:
            tree_new = tree_new +tree[i]
        i +=1
    tree_new = tree_new[tree_new.find('=')+1:]
    return tree_new

def create_random_tree(labels,blens,outgroup=None):
    nodes = labels[:]
    if outgroup != None:
        if outgroup not in nodes:
            print("Warning: outgroup was not correctly specified")
            sys.exit(-1)
    print("@",len(blens),file=sys.stderr)
    biter = iter(blens)
    #print(biter)
    rng = default_rng()
    #print(nodes)
    #print("@@@@@@@@",labels)
    #sys.exit()
    #print(len(blens))
    if outgroup != None:
        i = nodes.index(outgroup)
        nodes.pop(i)
    counter = 0
    while len(nodes)>1:
        a,b = rng.permutation(range(len(nodes)))[:2] #rng.integers(low=0, high=len(nodes), size=2)
        la,lb = next(biter),next(biter)
        # [(a:4,b:2),c,d,e]
        # [((a:4,b:2):4,e:2),c,d]
        c = f'({nodes[a]}:{la:.10f},{nodes[b]}:{lb:.10f})'
        counter += 2
        nodes[a] = c
        nodes.pop(b)
        #print("@nodelen",len(nodes))
    #print(nodes)
    if outgroup != None:
        la,lb = next(biter),next(biter)
        c = f'({nodes[0]}:{la:.10f},{outgroup}:{lb:.10f})'
        nodes[0] = c
        counter += 2
    #sys.exit()
    print("@",counter,file=sys.stderr)
    return nodes[0]+":0.0;"
    
def generate_random_tree(labels,totaltreelength,outgroup=None):
    rng = default_rng()
    a = [1]*(-2 + len(labels)*2)
    blen = totaltreelength * rng.dirichlet(a)
    rt = create_random_tree(labels, blen.tolist(), outgroup)
    #print(rt)
    #sys.exit()
    return rt
    
if __name__ == "__main__":
    import time
    nelder = True
        #    import data
    if "--test" in sys.argv:
        import phylip as ph
        infile = sys.argv[2]
        starttrees = sys. argv[3]
        labels, sequences, variable_sites = ph.readData(infile)
        with open(starttrees,'r') as f:
            StartTrees = [line.strip() for line in f]
    
        print('\n\n\nNelder-Mead optimization')
        for newick in StartTrees:
            mtree = Tree()
            mtree.myread(newick,mtree.root)
            mtree.insertSequence(mtree.root,labels,sequences)
            tic = time.perf_counter()
            original = mtree.likelihood()
            toc = time.perf_counter()
            print("Likelihood calculation\nTimer:", toc-tic)     
            print("using downpass algorithm:", original)
            delegate = []
            print()
            tic = time.perf_counter()
            mtree.root.name='root'
            mtree.delegate_extract(mtree.root,delegate)
            x = mtree.delegate_calclike(delegate)
            toc = time.perf_counter()
            print("Timer:", toc-tic,"\nusing delegate algorithm:",x)     
            print('\n\n\nNelder-Mead optimization')
            tic = time.perf_counter()
            newtree = mtree.optimize()
            toc = time.perf_counter()
            newtree.likelihood()
            print("Nelder_mead optimization:\nTimer:", toc-tic,"\nlnL=",newtree.lnL)
    else:
        newick = "(0BAA:0.0564907002,(0BAB:0.0060581140,(0BAC:0.0025424508,0BAD:0.0025424508):0.0035156632):0.0104325862):0.0000000000"
        labels = ["0BAA","0BAB","0BAC","0BAD"]
        sequences = ['AA','GA','CA','CC']
        mtree = Tree()
        mtree.myread(newick,mtree.root)
        mtree.insertSequence(mtree.root,labels,sequences)
        tic = time.perf_counter()
        original = mtree.likelihood()
        toc = time.perf_counter()
        print("Likelihood calculation\nTimer:", toc-tic)     
        print("using downpass algorithm:", original)
        delegate = []
        print()
        tic = time.perf_counter()
        mtree.root.name='root'
        mtree.delegate_extract(mtree.root,delegate)
        x = mtree.delegate_calclike(delegate)
        toc = time.perf_counter()
        print("Timer:", toc-tic,"\nusing delegate algorithm:",x)     

        tic = time.perf_counter()
        newtree = mtree.scioptimize()
        toc = time.perf_counter()
        print(newtree.lnL)
        lnl = newtree.likelihood()
        print("\n\n\nNelder_mead optimization:\nTimer:", toc-tic,"\nlnL=",lnl)
        m2tree = Tree()
        m2tree.myread(newick,m2tree.root)
        m2tree.insertSequence(m2tree.root,labels,sequences)

        tic = time.perf_counter()
        newtree12 = m2tree.optimizeNR()
        newtree22 = m2tree.optimizeNR()
        newtree32 = m2tree.optimizeNR()
        newtree42 = m2tree.optimizeNR()
        toc = time.perf_counter()
        #newtree2.likelihood()
        print("Newton-Raphson optimization:\nTimer:", toc-tic,"\nlnL=",newtree12,newtree22,newtree32,newtree42)
        #newtree.myprint(newtree.root)
        #newtree2.myprint(newtree2.root)
        
