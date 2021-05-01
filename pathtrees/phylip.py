# functions to read a phylip sequence file and to read a NEWICK tree string
# -readData      : read the phylip file and returns 
#                  labels and sequences as lists
# -readTreeString: read NEWICK formated trees from a file 
#                  and returns a list of strings 
# PB Oct 2011, PB,MK April 2021
DEBUG = False
def readData(file):     #testdata.phy : sequences
    f = open(file,'r')
    label =[]
    sequence=[]
    data = f.readlines()
    f.close()
    numind,numsites = (data.pop(0)).split()
    for i in data:
        if i=='':
            continue
        l = i[:10]    #this assumes standard phylip format
        s = i[11:]    #
        label.append(l.strip())
        sequence.append(s.strip())
    if DEBUG:
        print ("Phylip file:", file)
        print ("    species:", numind)
        print ("    sites:  ", numsites)
    varsites = [list(si) for si in sequence if len(si)>0]
    #print(len(varsites),len(varsites[0]))
    varsites = [len([i for i in list(set(si)) if i!='-']) for si in zip(*varsites)]
    varsites = [sum([vi>1 for vi in varsites]),len(varsites)] 
    #print(varsites)
    return label,sequence,varsites

def readTreeString(file):     #testdata.tre : Newick trees
    f = open(file,'r')
    data = f.readlines()
    f.close()
    return data

def guess_totaltreelength(sites):
    return sites[0]/sites[1]  #aka p-distance this may need a lot of improvement



