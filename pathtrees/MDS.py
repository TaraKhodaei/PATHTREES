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
    idx=sort_index[-10:]    #10 best
    Like_Big = [Likelihood[i] for i in idx]
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

def interpolate_grid(filename, N,n, M,Likelihood, bestlike, StartTrees):
    meth= 'cubic'
    num=100
    X= MDS(M,2)
    xx = np.linspace(np.min(X),np.max(X),num)
    yy =  np.linspace(np.min(X),np.max(X),num)
    XX, YY = np.meshgrid(xx,yy)          #We DONT know the likelihood values of these points
    values = griddata((np.real(X[:,0]), np.real(X[:,1])), Likelihood, (XX, YY), method=meth)

    fig = plt.figure(figsize=(12,5))

    ax1 = plt.subplot2grid((15,70), (3,0),rowspan=8, colspan=15)
    contour = ax1.contourf(XX, YY, values, alpha=0.9, vmin=np.nanmin(values), vmax=np.nanmax(values))
    points = ax1.scatter(np.real(X[:-N,0]), np.real(X[:-N,1]), c=Likelihood[:-N] ,alpha=1, marker='^', cmap='viridis',  edgecolors="black", linewidths=0.5, s=40)     #starttrees
    ax1.scatter(X[-N:,0], X[-N:,1], alpha=0.8,  marker='o', c=Likelihood[-N:] , cmap='viridis',  edgecolors="black", linewidths=0.5, s=7)  #PathTrees
    for i in range(len(StartTrees)):     #starttrees_names
        ax1.text(X[i,0],X[i,1],i, color='black', size=7, fontweight="bold")
    idx = list(zip(*bestlike))[0]
    print("idx =", idx)
    for i in idx:      #best trees
        ax1.scatter(X[i,0],X[i,1],c='r',s=12)
        ax1.text(X[i,0],X[i,1],i,color='black', size=7,)
    ax1.grid(linestyle=':', linewidth='0.3', color='black')
    #ax1.set_xlim(np.min(X[:,0])-0.01, np.max(X[:,0])+0.01)
    #ax1.set_ylim(np.min(X[:,1])-0.01, np.max(X[:,1])+0.01)
    ax1.set_xlim(np.min(X[:,0]), np.max(X[:,0]))
    ax1.set_ylim(np.min(X[:,1]), np.max(X[:,1]))
    ax1.set_xlabel('Coordinate 1')
    ax1.set_ylabel('Coordinate 2')


    ax2 = plt.subplot2grid((15,70), (0,15), rowspan=15, colspan=70, projection='3d')
    ax2.plot_wireframe(XX, YY, values,0.5, cmap='viridis')
    #NOTE: remove nans by vmin=np.nanmin(values), vmax=np.nanmax(values))
    surf = ax2.plot_surface(XX, YY, values, rstride=1, cstride=1,
                            edgecolor='none',alpha=0.5, cmap='viridis',vmin=np.nanmin(values), vmax=np.nanmax(values))
    cset = ax2.contourf(XX, YY, values, vmin=np.nanmin(values), vmax=np.nanmax(values),
                            zdir='z', offset=np.min(Likelihood)-15, cmap='viridis', alpha=0.4)
    points = ax2.scatter3D(X[:-N,0], X[:-N,1], Likelihood[:-N], alpha = 1, marker='^', c=Likelihood[:-N] , cmap='viridis',  edgecolors="black", linewidths=0.5, s=35)    #starttrees
    ax2.scatter3D(X[-N:,0],X[-N:,1],Likelihood[-N:], alpha = 0.8, marker='o', c=Likelihood[-N:] , cmap='viridis',  edgecolors="black", linewidths=0.5, s=7)   #PathTrees
    for i in range(len(StartTrees)):     #starttrees_names
        ax2.text(X[i,0],X[i,1],Likelihood[i],i, color='black', size=7, fontweight="bold")
    for i in idx:      #best trees
        ax2.scatter3D(X[i,0],X[i,1],Likelihood[i],c='r',s=8)
        ax2.text(X[i,0],X[i,1],Likelihood[i],i,color='black', size=7)

    ax2.grid(linestyle='-', linewidth='5', color='green')
    cbar = fig.colorbar(points, ax = ax2, shrink = 0.6, aspect = 12)
    cbar.set_label('Log Likelihood', rotation=270 , labelpad=15)
    ax2.set_xlabel('Coordinate 1', labelpad=10)
    ax2.set_ylabel('Coordinate 2', labelpad=10)
    ax2.set_zlabel('Coordinate 3', labelpad=10)
    ax2.set_xlim(np.min(X[:,0]), np.max(X[:,0]))
    ax2.set_ylim(np.min(X[:,1]), np.max(X[:,1]))

    fig.tight_layout()
    plt.savefig(filename)
    #,'Contour_Surface_{}.png'.format(meth), format='png')
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
