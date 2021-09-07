import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.vq import vq, kmeans, whiten

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
import statsmodels.api as sm
import sys

from scipy.interpolate import griddata
from mpl_toolkits import mplot3d

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Reading Files~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_from_files(likelihoodfile,treefile,pathtreefile,starttreefile):
    like = open(likelihoodfile, "rb")
    like = like.readlines()
    Likelihood = np.loadtxt(like)
    print("length of Likelihood= ", len(Likelihood))

    tree = open(treefile, "r") 
    treelist = tree.readlines()
    print("length of treelist : ", len(treelist))

    Path = open(pathtreefile, "rb")
    pathlist = Path.readlines()
    N= len(pathlist)
    print("N= ", len(pathlist))
    
    start = open(starttreefile, "r")
    StartTrees = start.readlines()
    print("length of StartTrees : ", len(StartTrees))
    
    return Likelihood,treelist,pathlist,StartTrees
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_GTP_distances(n,GTPOUTPUT):
    data=[]
    File = open(GTPOUTPUT, "r")
    file = File.readlines()
    for line in file:
        temp=line.rstrip().split('\t')
        data.append(temp[-1])
    data = [x for x in data if x != '']
    ind = np.triu_indices(n,k=1)
    M = np.zeros((n,n))
    M[ind]=data
    for i in range(len(M)):
        for j in range(len(M)):
            M[j][i]=M[i][j]
    distances=M
    return distances

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MDS Algorithm~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def MDS(M,d):
    n=len(M)
    print("n=",n)
    H=np.eye(n)-(1/n)*(np.ones((n,n)))
    K=-(1/2)*H.dot(M**2).dot(H)
    eval, evec = np.linalg.eig(K)
    idx = np.argsort(eval)[::-1] 
    eval = eval[idx][:d]
    evec = evec[:,idx][:, :d]
    X = evec.dot(np.diag(np.sqrt(eval)))
    X=X.real     #to get rid of imaginary part
    return X

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~likelihood index~~~~~~~~~~~~~~~~~~~~~~~
def best_likelihoods(Likelihood):
    #Note : we did not consider likelihood for PathTrees
    #sort=sorted(Likelihood)
    sort_index= sorted(range(len(Likelihood)), key=lambda k: Likelihood[k])

    if not proptype:
        m =  NUMBESTTREES #either n best likelihood trees or top 1/3 best likelihood tree
    else:
        m = int(2 * len(Likelihood)/3)
        
    idx=sort_index[-m:]    #10 best
    Like_Big = [Likelihood[i] for i in idx]
    print("Number in best trees list = ",len(idx))
    print("Best trees = ",idx)
    print("Best Likelihoods = ", Like_Big)
    return list(zip(idx,Like_Big))

def bestNstep_likelihoods(Likelihood, n, step):
    #Note : we did not consider likelihood for PathTrees
    #sort=sorted(Likelihood)
    lenL = len(Likelihood)
    if n > lenL:
        n=lenL
    sort_index= sorted(range(len(Likelihood)), key=lambda k: Likelihood[k])
    idx=sort_index[-n::step]    #10 best
    Like_Big = [Likelihood[i] for i in idx]
    print("Picked trees = ",idx)
    print("Picked Likelihoods = ", Like_Big)
    return list(zip(idx,Like_Big))


def plot_MDS(plotfile, N, n, M,Likelihood, bestlike, treelist, pathlist):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~2D MDS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X= MDS(M,2)
    fig = plt.figure(figsize=(12,5))
    axes=[None,None]

    fig.set_facecolor('white')
    plt.rcParams['grid.color'] = "white" # change color
    plt.rcParams['grid.linewidth'] = 1   # change linwidth
    plt.rcParams['grid.linestyle'] = '-'

    axes[0] = fig.add_subplot(1,2,1)
    s = axes[0].scatter(X[:-N,0], X[:-N,1],marker='^', c=Likelihood[:-N] , cmap='viridis', s=25)  #Not PathTrees
    axes[0].scatter(X[-N:,0], X[-N:,1], c=Likelihood[-N:] , cmap='viridis', s=6)  #PathTrees
    cbar = fig.colorbar(s)
    cbar.set_label('Log Likelihood', rotation=270 , labelpad=15)
    axes[0].set_xlabel('Coordinate 1')
    axes[0].set_ylabel('Coordinate 2')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~3D MDS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    axes[1] = fig.add_subplot(1,2,2,projection='3d')
    X1= MDS(M,3)
    axes[1].scatter(np.real(X1[:-N,0]),np.real(X1[:-N,1]),np.real(X1[:-N,2]), marker='^',c=Likelihood[:-N] , alpha=0.8, cmap='viridis', s=25)#Not PathTrees
    axes[1].scatter(np.real(X1[-N:,0]),np.real(X1[-N:,1]),np.real(X1[-N:,2]),c=Likelihood[-N:] , alpha=0.8, cmap='viridis', s=6)#PathTrees
    print("Best Trees:\n")
    idx = list(zip(*bestlike))[0]
    for i in idx:
        axes[1].scatter(np.real(X1[i,0]),np.real(X1[i,1]),np.real(X1[i,2]),c='r',s=10)
        axes[1].text(np.real(X1[i,0]),np.real(X1[i,1]),np.real(X1[i,2]),i,size=7)
        print("tree #",i,"\n",pathlist[i])
    axes[1].set_xlabel('Coordinate 1')
    axes[1].set_ylabel('Coordinate 2')
    axes[1].set_zlabel('Coordinate 3')
    plt.tight_layout()
    plt.savefig(plotfile)
    #plt.show()


#~~~~~~~~~~~~~~~~~~~interpolation_griddata (Contour & Surface)~~~~~~~~~~~~~~~~~~~~~~
#meth: linear, cubic, nearest

def interpolate_grid(filename, N, n, M, Likelihood, bestlike, StartTrees, hull_indices=None):
    meth= 'cubic'
    num=100
    numstarttrees = len(StartTrees)
    X= MDS(M,2)
    xx = np.linspace(np.min(X),np.max(X),num)
    yy =  np.linspace(np.min(X),np.max(X),num)
    XX, YY = np.meshgrid(xx,yy)          #We DONT know the likelihood values of these points
    values = griddata((np.real(X[:,0]), np.real(X[:,1])), Likelihood, (XX, YY), method=meth)

    fig = plt.figure(figsize=(12,5))

    fig.set_facecolor('white')
    plt.rcParams['grid.color'] = "white" # change color
    plt.rcParams['grid.linewidth'] = 1   # change linwidth
    plt.rcParams['grid.linestyle'] = '-'

    ax1 = plt.subplot2grid((20,342), (7,0),rowspan=7, colspan=80)
    contour = ax1.contourf(XX, YY, values, 10, alpha=1, vmin=np.nanmin(values), vmax=np.nanmax(values), cmap='viridis')

    #data range
    vmin, vmax = min(Likelihood), max(Likelihood)    #To keep consistency between colors, use the vmin, vmax arguments when breaking up the points to use colormap and different markers for different values
    
    if it==0:
        ax1.scatter(X[:N,0], X[:N,1], c=Likelihood[:N] ,alpha=1, marker='^', vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.5, s=35, label=f"{numstarttrees} Starting trees")     #starttrees
    if it>0:
        ax1.scatter(X[:N-1,0], X[:N-1,1], c=Likelihood[:N-1] ,alpha=1, marker='^', vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.5, s=35, label=f"{numstarttrees} Starting trees")     #starttrees without previous best tree
        ax1.scatter(X[N-1,0], X[N-1,1], c=Likelihood[N-1] ,alpha=1, marker='^', vmin = vmin, vmax = vmax, cmap='viridis',      edgecolors="black", linewidths=1.2, s=80, label="Previous best tree \nLogLike {:4.4f}".format(Likelihood[N-1]))   #best tree from the previous iteration

    points = ax1.scatter(X[N:,0], X[N:,1], alpha=0.8,  marker='o', c=Likelihood[N:] , cmap='viridis',  edgecolors="black", linewidths=0.3, s=5, label="All {} Path trees".format(len(Treelist)))  #PathTrees

    idx = list(zip(*bestlike))[0]
    ax1.scatter(X[idx[:-1],0],X[idx[:-1],1],c='r',s=6, label="{} trees with highest likelihood".format(len(idx)))      #best trees

    if hull_indices != None:
         ax1.scatter(X[hull_indices,0],X[hull_indices,1], marker='o',c='r', edgecolors="black", linewidths=0.6, s=10, label="{} boundary trees".format(len(hull_indices)))   #boundary trees

    
    t = idx[-1]
    ax1.scatter(X[t,0],X[t,1], marker='o',c='r', edgecolors="black", linewidths=1, s=60, label="Current best tree \nLogLike {:4.4f}".format(Likelihood[t]))

    
    min_xvalue, max_xvalue = ax1.get_xlim()       #To hadle the distance between boundary and contour
    min_yvalue, max_yvalue = ax1.get_ylim()
    x_dist =(max_xvalue-min_xvalue)/30
    y_dist =(max_yvalue-min_yvalue)/30

    ax1.grid(linestyle=':', linewidth='0.1', color='grey', alpha=0.6)
    ax1.set_xlim(np.min(X[:,0])-x_dist, np.max(X[:,0])+x_dist)
    ax1.set_ylim(np.min(X[:,1])-y_dist, np.max(X[:,1])+y_dist)
    ax1.set_xlabel('Coordinate 1', labelpad=3, fontsize=8)
    ax1.set_ylabel('Coordinate 2', labelpad=3, fontsize=8)
    ax1.tick_params(labelsize=4.5)
    
    ax1.ticklabel_format(useOffset=False, style='plain')
    fig.autofmt_xdate()    #to fix the issue of overlapping x-axis labels
    #    plt.xticks(rotation=30)

    plt.legend(bbox_to_anchor=(1.05, 0.84),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1)

    ax2 = plt.subplot2grid((20,342), (0,142), rowspan=20, colspan=200, projection='3d')
    ax2.plot_wireframe(XX, YY, values,0.25, cmap='viridis')
    #NOTE: remove nans by vmin=np.nanmin(values), vmax=np.nanmax(values))
    surf = ax2.plot_surface(XX, YY, values, rstride=1, cstride=1,
                            edgecolor='none',alpha=0.30, cmap='viridis',vmin=np.nanmin(values), vmax=np.nanmax(values))
    min_value, max_value = ax2.get_zlim()       #To hadle the distance between surface and contour
    cset = ax2.contourf(XX, YY, values, vmin=np.nanmin(values), vmax=np.nanmax(values),
                        zdir='z', offset=min_value, cmap='viridis', alpha=0.3)

    if it==0:
        ax2.scatter3D(X[:N,0], X[:N,1], Likelihood[:N], alpha = 1, marker='^', c=Likelihood[:N] , vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.5, s=35)    #starttrees
    if it>0:
        ax2.scatter3D(X[:N-1,0], X[:N-1,1], Likelihood[:N-1], alpha = 1, marker='^', c=Likelihood[:N-1] ,vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.5, s=35)    #starttrees without previous best tree
        ax2.scatter3D(X[N-1,0], X[N-1,1], Likelihood[N-1], alpha = 1, marker='^',  c=Likelihood[N-1] , vmin = vmin, vmax = vmax, cmap='viridis', edgecolors="black", linewidths=1.2, s=80)    #best tree from the previous iteration

    points = ax2.scatter3D(X[:,0],X[:,1],Likelihood, alpha = 0.8, marker='o', c=Likelihood , cmap='viridis',  edgecolors="black", linewidths=0.3, s=6)   #PathTrees
    
    for num, i in enumerate(idx[:-1]):      #best trees
        ax2.scatter3D(X[i,0],X[i,1],Likelihood[i],c='r',s=12)

    t = idx[-1]    #current best tree
    ax2.scatter3D(X[t,0],X[t,1],Likelihood[t], marker='o',c='r', edgecolors="black", linewidths=1, s=65)
    print("best current tree #",t,"\n",Treelist[t])

    ax2.grid(linestyle='-', linewidth='5', color='green')
    v1 = np.linspace(np.min(Likelihood), np.max(Likelihood), 7, endpoint=True)
    cbar = fig.colorbar(points, ax = ax2, shrink = 0.32, aspect = 15, pad=0.11, ticks=v1)
    cbar.ax.set_yticklabels(["{:4.4f}".format(i) for i in v1])    #To get rid of exponention expression of cbar numbers

    cbar.set_label('Log Likelihood', rotation=270 , labelpad=15, fontsize=8)
    ax2.set_xlabel('Coordinate 1', labelpad=1,  fontsize=8)
    ax2.set_ylabel('Coordinate 2', labelpad=2,  fontsize=8)
    ax2.set_zlabel('Coordinate 3', labelpad=8,  fontsize=8)
    ax2.set_xlim(np.min(X[:,0]), np.max(X[:,0]))
    ax2.set_ylim(np.min(X[:,1]), np.max(X[:,1]))
    
    ax2.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
    ax2.tick_params(axis='z',labelsize=5 , pad=4)    #to change the size of numbers on axis & space between ticks and labels

    cbar.ax.tick_params(labelsize=6)   #to change the size of numbers on cbar

    ax2.ticklabel_format(useOffset=False, style='plain')   # to get rid of exponential format of numbers on all axis
    plt.ticklabel_format(useOffset=False)    # to get rid of exponential format of numbers on all axis
    ax2.view_init(21, -44)     #change the angle of 3D plot
    fig.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=-0.25, right=0.98, top=1.3, wspace=None, hspace=None)  #to adjust hite spaces around figure
    plt.savefig(filename)
    #plt.show()  





if __name__ == "__main__":
    GTPOUTPUT = 'results/output.txt'
    likelihoodfile = "results/likelihoods"
    treefile = "results/treelist"
    pathtreefile = "results/pathTrees"
    starttreefile = "results/startTrees"
    Likelihood,treelist,pathlist, StartTrees = read_from_files(likelihoodfile,treefile,pathtreefile, starttreefile)

    bestlike = best_likelihoods(Likelihood)
    n = len(treelist)
    N= len(pathlist)
    distances = read_GTP_distances(n,GTPOUTPUT)
    file = "TESTPLOT.pdf"
    plot_MDS(file,N,n,distances, Likelihood, bestlike, treelist, pathlist)
    interpolate_grid(distances,Likelihood, bestlike, StartTrees)
