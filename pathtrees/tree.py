#!/usr/bin/env python3


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
import likelihood as like

#=========================================  Peter code  ==========================================

def getName(s):
    name = ""
    for j in range(len(s)):
        if((s[j]==')') | (s[j]=='(') | (s[j]==':') | (s[j]==' ') | (s[j]==',')):
            return j,name
        else:
            name= name + s[j]
    return j,name



def istip(p):
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
        if(newick[self.i]=="("):
            q = Node()
            p.left = q
            self.i += 1
            self.myread(newick,q)
            if(newick[self.i]!=','):
                print ("error reading, failed to find ',' in %s" % newick)
            self.i += 1
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
            pos = label.index(p.name.strip())
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

def create_random_tree(labels,blens):
    nodes = labels[:]
    #print(blens)
    biter = iter(blens)
    #print(biter)
    rng = default_rng()
    print(nodes)
    while len(nodes)>1:
        a,b = rng.permutation(range(len(nodes)))[:2] #rng.integers(low=0, high=len(nodes), size=2)
        la,lb = next(biter),next(biter)
        c = f'({nodes[a]}:{la:.10f},{nodes[b]}:{lb:.10f})'
        nodes[a] = c
        nodes.pop(b)
        #print(nodes)
    #print(nodes)
    #sys.exit()
    return nodes[0]+":0.0;"
    
def generate_random_tree(labels,totaltreelength):
    rng = default_rng()
    a = [1]*(len(labels)*2)
    blen = rng.dirichlet(a)
    rt = create_random_tree(labels, blen.tolist())
    return rt
    
