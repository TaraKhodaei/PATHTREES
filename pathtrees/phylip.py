# functions to read a phylip sequence file and to read a NEWICK tree string
# -readData      : read the phylip file and returns 
#                  labels and sequences as lists
# -readTreeString: read NEWICK formated trees from a file 
#                  and returns a list of strings 
# PB Oct 2011
def readData(file):     #testdata.phy : sequences
    f = open(file,'r')
    label =[]
    sequence=[]
    data = f.readlines()
    f.close()
    numind,numsites = (data.pop(0)).split()
    for i in data:
        l = i[:10]    #Tara
        s = i[11:]    #Tara
#        l = i[:32]   #Tara
#        s = i[33:]   #Tara
        label.append(l.strip())
        sequence.append(s.strip())
#    print ("Phylip file:", file)
#    print ("    species:", numind)
#    print ("    sites:  ", numsites)
    return [label,sequence]

def readTreeString(file):     #testdata.tre : Newick trees
    f = open(file,'r')
    data = f.readlines()
    f.close()
    return data
