#!/usr/bin/env python
#
#
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent
sys.path.append(str(file.parent))

#print(sys.path)
import tree

from io import StringIO

#from the internet, needs reference
class Redirectedstdout:
    def __init__(self):
        self._stdout = None
        self._string_io = None

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._string_io = StringIO()
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout

    def __str__(self):
        return self._string_io.getvalue()
    

def print_newick_string(tips,edges,tiplen,edgelen):
    if len(edges)==0:     #Tara did this for debuging
        newick  = '('+str(tips[0])+':'+str(tiplen[0])+','+ str(tips[1])+':'+str(tiplen[1])+')'+':0.0'
        return  newick  # this should probably abort!
    treenodes = {}
    for name,blen in zip(tips,tiplen):
        p = tree.Node()
        p.name = name
        p.blength = blen
        treenodes[name] = p

    numelem = [len(xi) for xi in edges]
    x = list(zip(edges, edgelen, numelem))
    x = sorted(x,key=lambda x: x[2] )
    for e, el, elem in x:
        if elem == 2:
            i = tree.Node()
            i.name = "|".join(list(sorted(e)))
            i.blength = el
            i.left = treenodes[e[0]]
            i.right = treenodes[e[1]]
            treenodes[e[0]].ancestor = i
            treenodes[e[1]].ancestor = i
            treenodes[i.name] = i
            del treenodes[e[1]]
            del treenodes[e[0]]
            continue
        elif elem > 2:
            i = tree.Node()
            i.name = "|".join(list(sorted(e)))
            i.blength = el
            pick =[]
            #print("i.name",i.name)
            for key in treenodes:
                if key in i.name:
                    pick.append(key)
            if len(pick)==0:
                for key in treenodes:
                    keylist = key.split('|')
                    keylen = len(keylist)
                    c=0
                    for k in keylist:
                        if k in i.name:
                            c += 1
                    if c==keylen:     
                        pick.append(key)
            if len(pick)==1:
                tempstr = i.name
                tempstr = tempstr.replace(pick[0],"").replace("||","|")
                for key in treenodes:
                    if key in tempstr:
                        pick.append(key)
            if len(pick)==2:
                i.left = treenodes[pick[0]]
                i.right = treenodes[pick[1]]
                treenodes[pick[0]].ancestor = i
                treenodes[pick[1]].ancestor = i
                treenodes[i.name] = i
                del treenodes[pick[1]]
                del treenodes[pick[0]]
            else:
                print("works only for bifurcating trees")
                for di in treenodes:
                    print("DICT",di)
                sys.exit()
    # we should be back down to two treenodes entries
    q = tree.Node()
    q.name = 'root'
    keys = list(treenodes.keys())
    n1 = treenodes[keys[0]]
    n2 = treenodes[keys[1]]
    q.left = n1
    q.right = n2
    n1.ancestor = q
    n2.ancestor = q
    q.blength=0.0
    t = tree.Tree()
    t.root=q
    t.remove_internal_labels(t.root)
    with Redirectedstdout() as newick:
        t.myprint(t.root,file=sys.stdout)    
    return str(newick)

#if __name__ == "__main__":

 
#    edges = [['Microcebus_murinus', 'Cheirogaleus_major'], ['Propithecus_coquereli', 'Lemur_catta', 'Varecia_variegata_variegata'], ['Lemur_catta', 'Varecia_variegata_variegata']]
#
#    edgelengths = [0.050346, 0.023932, 0.042509]
#
#    tips = ['Propithecus_coquereli', 'Lemur_catta', 'Varecia_variegata_variegata', 'Microcebus_murinus', 'Cheirogaleus_major']
#    tipslength = [0.085616, 0.075789, 0.093902, 0.104756, 0.06542]
#
#    newick = print_newick_string(tips,edges,tipslength, edgelengths)
#    print(newick)
#
#    
#    edges = [['Cheirogaleus_major', 'Microcebus_murinus', 'Lemur_catta', 'Varecia_variegata_variegata'], ['Lemur_catta', 'Varecia_variegata_variegata'], ['Cheirogaleus_major', 'Microcebus_murinus']]
#    edgelengths = [0.022699, 0.03823, 0.032412]
#    tips = ['Cheirogaleus_major', 'Microcebus_murinus', 'Lemur_catta', 'Varecia_variegata_variegata', 'Propithecus_coquereli']
#    tipslength = [0.087437, 0.118768, 0.094438, 0.119288, 0.105318]
#    
#    newick = print_newick_string(tips,edges,tipslength, edgelengths)
#    print(newick)


    
