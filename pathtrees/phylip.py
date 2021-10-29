# functions to read a phylip sequence file and to read a NEWICK tree string
# -readData      : read the phylip file and returns 
#                  labels and sequences as lists
# -readTreeString: read NEWICK formated trees from a file 
#                  and returns a list of strings 
# PB Oct 2011, PB,MK April 2021
DEBUG = False
def readData(myfile, type='STANDARD'):     #testdata.phy : sequences
    f = open(myfile,'r')
    label =[]
    sequence=[]
    data = f.readlines()
    f.close()
    numind,numsites = (data.pop(0)).split()
    for i in data:
        if i=='':
            continue
        if type=='STANDARD':
            l = i[:10]    #this assumes standard phylip format
            #print("myread()",l)
            s = i[11:]    #
        else:
            index = i.rfind('  ')
            l = i[:index]
            s = i[index+1:]
            
        label.append(l.strip().replace(' ','_'))
        #print("myread()",l.replace(' ','_').strip())
        sequence.append(s.strip())
    if DEBUG:
        print ("Phylip file:", myfile, file=sys.stderr)
        print ("    species:", numind, file=sys.stderr)
        print ("    sites:  ", numsites, file=sys.stderr)
        print ("first label:",label[0], file=sys.stderr)
        print ("last  label:",label[-1], file=sys.stderr)
    varsites = [list(si) for si in sequence if len(si)>0]
    #print(len(varsites),len(varsites[0]))
    varsites = [len([i for i in list(set(si)) if i!='-']) for si in zip(*varsites)]
    varsites = [sum([vi>1 for vi in varsites]),len(varsites)] 
    #print(varsites)
    #print(label)
    return label,sequence,varsites

def readTreeString(myfile):     #testdata.tre : Newick trees
    f = open(myfile,'r')
    data = f.readlines()
    f.close()
    return data

def guess_totaltreelength(sites):
    return sites[0]/sites[1]  #aka p-distance this may need a lot of improvement



