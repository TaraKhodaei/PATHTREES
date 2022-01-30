import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
#from scipy.cluster.vq import vq, kmeans, whiten

#from sklearn.cluster import KMeans
#from sklearn.cluster import DBSCAN


import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 100000
import statsmodels.api as sm
import sys

from scipy.interpolate import griddata
from mpl_toolkits import mplot3d


import matplotlib.lines as mlines
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d

from scipy.interpolate import Rbf

#paup_tree = False    #first data option
#paup_MAP = True    #first data option
#paup_tree = True    #second data option
#paup_MAP = False    #second data option
paup_tree = False    #No paup
paup_MAP = False    #No paup&MAp
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
#    print("n=",n)
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
def best_likelihoods(Likelihood, n=20):
    # n is NUMBESTTREES from options
    sort_index= sorted(range(len(Likelihood)), key=lambda k: Likelihood[k])
    if n >= len(sort_index):
        idx=sort_index
    else:
        idx=sort_index[-n:]    #default is 10 best

#    Like_Big = [Likelihood[i] for i in idx]
#    print("Number in best trees list = ",len(idx))
#    print("Best trees = ",idx)
#    print("Best Likelihoods = ", Like_Big)
#    return list(zip(idx,Like_Big))
    return idx

def bestNstep_likelihoods(Likelihood, n, step):
    lenL = len(Likelihood)
    if n > lenL:
        n=lenL
    sort_index= sorted(range(len(Likelihood)), key=lambda k: Likelihood[k])
    idx=sort_index[-n::step]    # last m best, every step  ---> n=0 & step=1 : all trees
    Like_Big = [Likelihood[i] for i in idx]
#    print("Picked trees = ",idx)
#    print("Picked Likelihoods = ", Like_Big)
#    return list(zip(idx,Like_Big))
    return idx


def plot_MDS(plotfile, N, n, M,Likelihood, bestlike, treelist, pathlist):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~2D MDS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X= MDS(M,2)
    #print(X)
    #sprint(Likelihood)
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

def interpolate_grid(it, filename, M, Likelihood, bestlike, Treelist, StartTrees,  optimized_BestTrees, Topologies,NUMPATHTREES, hull_indices=None):
    
    meth= 'cubic'
    n = len(Treelist)
    N = len(StartTrees)
    n_path = NUMPATHTREES-2
    if paup_tree:
        opt = len(optimized_BestTrees)+1    #just PAUP
    elif paup_MAP:
        opt = len(optimized_BestTrees)+2    #both PAUP&MAP
    else:
        opt = len(optimized_BestTrees)
        
    all_path = n-opt-N

    num=200
    X= MDS(M,2)
    xx = np.linspace(np.min(X),np.max(X),num)
    yy =  np.linspace(np.min(X),np.max(X),num)
    XX, YY = np.meshgrid(xx,yy)          #We DONT know the likelihood values of these points
    values = griddata((np.real(X[:,0]), np.real(X[:,1])), Likelihood, (XX, YY), method=meth)

    N_topo = len(Topologies)
    colormap = plt.cm.RdPu
    Colors = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo)]

    fig = plt.figure(figsize=(12,5))
    fig.set_facecolor('white')
    plt.rcParams['grid.color'] = "white" # change color
    plt.rcParams['grid.linewidth'] = 1   # change linwidth
    plt.rcParams['grid.linestyle'] = '-'

    ax1 = plt.subplot2grid((40,345), (14,0),rowspan=13, colspan=80)
    contour = ax1.contourf(XX, YY, values, 10, alpha=1,
                           vmin=np.nanmin(values), vmax=np.nanmax(values), cmap='viridis')

    vmin, vmax = np.nanmin(values.flatten()), np.nanmax(values.flatten())
    #    print(f"vmin = {vmin} , vmax = {vmax}")
    
    ax1.scatter(X[:N,0], X[:N,1], alpha=1, marker='^',facecolors='none', vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.7, s=60)     #starttrees

    points = ax1.scatter(X[N:,0], X[N:,1], c=Likelihood[N:] , alpha=1,  marker='o', cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
    if paup_tree:
        ax1.scatter(X[-1,0],X[-1,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax1.scatter(X[-2,0],X[-2,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        opt_points = ax1.scatter(X[-opt:-1,0],X[-opt:-1,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    
    elif paup_MAP:
        ax1.scatter(X[-3,0],X[-3,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax1.scatter(X[-1,0],X[-1,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax1.scatter(X[-2,0],X[-2,1], marker='o', facecolors='black', edgecolors="black", linewidths=0.5, s=140)     #MAP
        opt_points = ax1.scatter(X[-opt:-2,0],X[-opt:-2,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    else:
        ax1.scatter(X[-1,0],X[-1,1], marker='o', facecolors='red', edgecolors="black", linewidths=0.5, s=200)     #best optimized
        opt_points = ax1.scatter(X[-opt:,0],X[-opt:,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        
    for j, (top, c) in enumerate(zip(Topologies, Colors)):     #topologies trees
        ax1.scatter(X[top,0], X[top,1], marker='o', color=c, edgecolors="black", linewidths=0.15,s=8)
    
    #    min_xvalue, max_xvalue = ax1.get_xlim()       #To hadle the distance between boundary and contour
    #    min_yvalue, max_yvalue = ax1.get_ylim()
    min_xvalue, max_xvalue = np.min(X[:,0]),np.max(X[:,0])       #To hadle the distance between boundary and contour
    min_yvalue, max_yvalue = np.min(X[:,1]),np.max(X[:,1])
    x_dist =(max_xvalue-min_xvalue)/30
    y_dist =(max_yvalue-min_yvalue)/30
    
    ax1.grid(linestyle=':', linewidth='0.2', color='white', alpha=0.6)
    #    ax1.set_xlim(min_xvalue, max_xvalue)
    #    ax1.set_ylim(min_yvalue, max_yvalue)
    ax1.set_xlim(min_xvalue-x_dist, max_xvalue+x_dist)
    ax1.set_ylim(min_yvalue-y_dist, max_yvalue+y_dist)
    
    ax1.set_xlabel('Coordinate 1', labelpad=7, fontsize=8)
    ax1.set_ylabel('Coordinate 2', labelpad=3, fontsize=8)
    ax1.tick_params(labelsize=4.5)
#    ax1.tick_params(color='k',  labelcolor='k', axis='both', which='both', labelsize=5, bottom=False, top=False, labelbottom=True,left=False, right=False, labelleft=True,  pad=0)      #to handle the visibility of ticks & distanc btw ticks and frame
#    ax1.set_frame_on(False)     #to handle the visibility of frame

    
    
    ax1.ticklabel_format(useOffset=False, style='plain')
    fig.autofmt_xdate()    #to fix the issue of overlapping x-axis labels
    
    #Labels:
    smallgreen_circle = mlines.Line2D([], [], color='limegreen', marker='o', linestyle='None',mec="black", mew=0.3, markersize=3, label="{} path + {} start trees:\n{}-trees between each pair".format(all_path,N, n_path))
    green_triangle = mlines.Line2D([], [], color='limegreen', marker='^', linestyle='None',mec="black", mew=0.4, markersize=8, label="{} starting trees".format(len(StartTrees)))
    smallpink_circle = mlines.Line2D([], [], color='hotpink', marker='o', linestyle='None', mec="black", mew=0.25, markersize=3, label="{} trees with highest likelihood ".format(len(bestlike)))
    bigpink_circle = mlines.Line2D([], [], color='hotpink', marker='o', linestyle='None',mec="black", mew=0.4, markersize=6, label="{} best optimized tree \n of different topologies".format(N_topo))
    
    
    if paup_tree:
        bigred_circle = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
        bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_circle, bigwhite_circle])
    
    elif paup_MAP:
        bigred_circle = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-3]))
        bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        bigblack_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='black', mew=0.9, markersize=11, label="MAP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
        plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_circle, bigwhite_circle, bigblack_circle])
    
    else:
        bigred_circle = mlines.Line2D([], [], color='red', marker='o', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        plt.legend(bbox_to_anchor=(1.05, 0.85),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_circle])
    plt.setp(ax1.spines.values(), color='gray')

    #=============================================================================

#    maximum_value = np.nanmin(values)
#
#    print(f"TEST : maximum_value = {maximum_value}")
#    values = values/maximum_value
#    Likelihood = Likelihood/maximum_value

    ax2 = plt.subplot2grid((40,345), (0,145), rowspan=50, colspan=200, projection='3d')
    ax2.plot_wireframe(XX, YY, values,  rstride=2, cstride=2, linewidth=1,alpha=0.3, cmap='viridis')
    surf = ax2.plot_surface(XX, YY, values,  rstride=1, cstride=1, edgecolor='none',alpha=0.5, cmap='viridis', vmin=np.nanmin(values), vmax=np.nanmax(values))
#    surf = ax2.plot_surface(XX, YY, values,  rstride=2, cstride=2, alpha=0.25, cmap='viridis',linewidth=1,  vmin=np.nanmin(values), vmax=np.nanmax(values))


    min_value, max_value = ax2.get_zlim()       #To hadle the distance between surface and contour

#    values = values/np.nanmax(values.flatten())
#    Likelihood = Likelihood/np.nanmax(values.flatten())
##    print(f"TEST : np.nanmax(values.flatten()) = {np.nanmax(values.flatten())}")





#    points = ax1.scatter(X[N:,0], X[N:,1], c=Likelihood[N:] , alpha=1,  marker='o', cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees

    cset = ax2.contourf(XX, YY, values, vmin=np.nanmin(values), vmax=np.nanmax(values), zdir='z', offset=min_value, cmap='viridis', alpha=0.3)

    ax2.scatter3D(X[:N,0], X[:N,1], Likelihood[:N], marker='^',facecolors='none',  vmin = vmin, vmax = vmax, edgecolors="black", linewidths=0.5, s=50)    #starttrees
    
    if paup_tree:
        ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax2.scatter3D(X[-2,0],X[-2,1],Likelihood[-2], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        opt_points2 = ax2.scatter3D(X[-opt:-1,0],X[-opt:-1,1],Likelihood[-opt:-1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    elif paup_MAP:
        
        ax2.scatter3D(X[-3,0],X[-3,1],Likelihood[-3], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax2.scatter3D(X[-2,0],X[-2,1],Likelihood[-2], marker='o',facecolors='black', edgecolors="black", linewidths=0.5, s=140)      #MAP
        opt_points2 = ax2.scatter3D(X[-opt:-2,0],X[-opt:-2,1],Likelihood[-opt:-2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        
    else:
        ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='red', edgecolors="black", linewidths=0.5, s=150)     #best optimized
        opt_points2 = ax2.scatter3D(X[-opt:,0],X[-opt:,1],Likelihood[-opt:], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        
    ax2.grid(linestyle='-', linewidth='5', color='green')
    points.set_clim([np.min(Likelihood), np.max(Likelihood)])

    
    ax2.set_xlabel('Coordinate 1', labelpad=1,  fontsize=8)
    ax2.set_ylabel('Coordinate 2', labelpad=2,  fontsize=8)
#    ax2.set_zlabel('Coordinate 3', labelpad=8,  fontsize=8)
#    ax2.set_zlabel('Log Likelihood', labelpad=6,  fontsize=8)
    ax2.set_xlim(np.min(X[:,0]), np.max(X[:,0]))
    ax2.set_ylim(np.min(X[:,1]), np.max(X[:,1]))
#    ax2.set_zlim(-1, 0)

    ax2.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
    ax2.tick_params(axis='z',labelsize=5 , pad=4)    #to change the size of numbers on axis & space between ticks and labels
    
    ax2.ticklabel_format(useOffset=False, style='plain')   # to get rid of exponential format of numbers on all axis
#    ax2.view_init(20, -45)     #change the angle of 3D plot
    ax2.view_init(20, 135)     #change the angle of 3D plot

    v1 = np.linspace(np.min(Likelihood), np.max(Likelihood), 7, endpoint=True)
    cbar = fig.colorbar(points, ax = ax2, shrink = 0.32, aspect = 12, pad=0.11, ticks=v1)
    cbar.ax.set_yticklabels(["{:4.4f}".format(i) for i in v1])    #To get rid of exponention expression of cbar numbers(main)
#    cbar.set_label('Log Likelihood', rotation=270 , labelpad=15, fontsize=8)
    cbar.set_label('Log Likelihood', rotation=90 , labelpad=-75, fontsize=8)
    cbar.ax.tick_params(labelsize=6)   #to change the size of numbers on cbar
    cbar.outline.set_edgecolor('none')
    plt.ticklabel_format(useOffset=False)

    #=============================================================================

    if paup_MAP:
        ax3 = plt.subplot2grid((40,345), (27,95), rowspan=1, colspan=45, frameon=False)
    else:
        ax3 = plt.subplot2grid((40,345), (25,95), rowspan=1, colspan=45, frameon=False)

    if paup_tree:     #extract paup one
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):-1]
        bounds = [bounds[0]-5]+ bounds
    elif paup_MAP:     #extract paup &MAP
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):-2]
        bounds = [bounds[0]-5]+ bounds
    else:
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):]
        bounds = [bounds[0]-5]+ bounds

    cmap = mpl.colors.ListedColormap(Colors_bar)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, spacing='uniform', orientation='horizontal')
    count=0
    for label in ax3.get_xticklabels():
        count +=1
        label.set_ha("right")
        label.set_rotation(40)

    ax3.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
    ax3.set_xticklabels(["{:4.4f}".format(i) for i in bounds])    #To get rid of exponention expression of numbers
    cb2.outline.set_edgecolor('white')   #colorbar externals edge color



    labels = ax3.get_xticklabels()
    len_label= len(labels)
    #    r = int(len_label/2)
#    r=8
    r=5
    print(f"======>len_label = {len_label}")
    if len(labels)>10:
        v2 = np.linspace(0,len_label, r,  dtype= int, endpoint=False)
        for i in range(len_label-1):
            if i not in v2:
                labels[i] = ""
    labels[0] = ""
    ax3.set_xticklabels(labels)



    fig.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=-0.25, right=0.98, top=1.3, wspace=None, hspace=None)  #to adjust white spaces around figure
    plt.savefig(filename)
    plt.show()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~interpolation_RBF (Contour & Surface)~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def interpolate_rbf(it, filename, M,Likelihood, bestlike, Treelist, StartTrees, optimized_BestTrees, Topologies, NUMPATHTREES):
    n = len(Treelist)
    N = len(StartTrees)
    n_path = NUMPATHTREES-2
    if paup_tree:
        opt = len(optimized_BestTrees)+1    #just PAUP
    elif paup_MAP:
        opt = len(optimized_BestTrees)+2    #both PAUP&MAP
    else:
        opt = len(optimized_BestTrees)
    
    all_path = n-opt-N
    num=200
    
    X= MDS(M,3)
    xx = np.linspace(np.min(X),np.max(X),num)
    yy =  np.linspace(np.min(X),np.max(X),num)
    zz =  np.linspace(np.min(X),np.max(X),num)
    np.savetxt ('MDS_X', X,  fmt='%s')
    
    
    XX, YY = np.meshgrid(xx,yy )
    rbfi = Rbf(np.real(X[:,0]), np.real(X[:,1]), Likelihood)
    values = rbfi(XX, YY)   # interpolated values(likelihods)

    rbf = Rbf(np.real(X[:,0]), np.real(X[:,1]), np.real(X[:,2]))
    ZZ = rbf(XX, YY)      # interpolated z coordinates
    
    N_topo = len(Topologies)
    colormap = plt.cm.RdPu
    Colors = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo)]
    
    fig = plt.figure(figsize=(12,5))
    fig.set_facecolor('white')
    plt.rcParams['grid.color'] = "white" # change color
    plt.rcParams['grid.linewidth'] = 1   # change linwidth
    plt.rcParams['grid.linestyle'] = '-'
    
    ax1 = plt.subplot2grid((40,345), (14,0),rowspan=13, colspan=80)
    contour = ax1.contourf(XX,YY,ZZ, 10, alpha=1, vmin=np.nanmin(ZZ), vmax=np.nanmax(ZZ), cmap='viridis')
#    contour = ax1.contourf(XX,YY,values, 10, alpha=1, vmin=np.nanmin(values), vmax=np.nanmax(values), cmap='viridis')
    
    
    vmin, vmax = np.nanmin(ZZ.flatten()), np.nanmax(ZZ.flatten())
#    vmin, vmax = np.nanmin(values.flatten()), np.nanmax(values.flatten())
#    print(f"vmin = {vmin} , vmax = {vmax}")

#    ax1.scatter(X[:N,0], X[:N,1], c=X[:N,2] ,alpha=1, marker='^', vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.7, s=60)     #starttrees
    ax1.scatter(X[:N,0], X[:N,1], alpha=1, marker='^',facecolors='none', vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.7, s=60)     #starttrees
    
#    points = ax1.scatter(X[N:,0], X[N:,1], alpha=0.8,  marker='o', c=Likelihood[N:] , cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
#    points = ax1.scatter(X[N:,0], X[N:,1], c=X[N:,2] , alpha=1,  marker='o', cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
    points = ax1.scatter(X[N:,0], X[N:,1], c=rbfi(X[N:,0],X[N:,1]) , alpha=1,  marker='o', cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
    if paup_tree:
        ax1.scatter(X[-1,0],X[-1,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax1.scatter(X[-2,0],X[-2,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        opt_points = ax1.scatter(X[-opt:-1,0],X[-opt:-1,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees

    elif paup_MAP:
        ax1.scatter(X[-3,0],X[-3,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax1.scatter(X[-1,0],X[-1,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax1.scatter(X[-2,0],X[-2,1], marker='o', facecolors='black', edgecolors="black", linewidths=0.5, s=140)     #MAP
        opt_points = ax1.scatter(X[-opt:-2,0],X[-opt:-2,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    else:
        ax1.scatter(X[-1,0],X[-1,1], marker='o', facecolors='red', edgecolors="black", linewidths=0.5, s=200)     #best optimized
        opt_points = ax1.scatter(X[-opt:,0],X[-opt:,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees

    for j, (top, c) in enumerate(zip(Topologies, Colors)):     #topologies trees
        ax1.scatter(X[top,0], X[top,1], marker='o', color=c, edgecolors="black", linewidths=0.15,s=8)
    
#    min_xvalue, max_xvalue = ax1.get_xlim()       #To hadle the distance between boundary and contour
#    min_yvalue, max_yvalue = ax1.get_ylim()
    min_xvalue, max_xvalue = np.min(X[:,0]),np.max(X[:,0])       #To hadle the distance between boundary and contour
    min_yvalue, max_yvalue = np.min(X[:,1]),np.max(X[:,1])
    min_zvalue, max_zvalue = np.min(X[:,2]),np.max(X[:,2])
    x_dist =(max_xvalue-min_xvalue)/30
    y_dist =(max_yvalue-min_yvalue)/30

    ax1.grid(linestyle=':', linewidth='0.2', color='white', alpha=0.6)
#    ax1.set_xlim(min_xvalue, max_xvalue)
#    ax1.set_ylim(min_yvalue, max_yvalue)
    ax1.set_xlim(min_xvalue-x_dist, max_xvalue+x_dist)
    ax1.set_ylim(min_yvalue-y_dist, max_yvalue+y_dist)

    ax1.set_xlabel('Coordinate 1', labelpad=7, fontsize=8)
    ax1.set_ylabel('Coordinate 2', labelpad=3, fontsize=8)
#    ax1.tick_params(labelsize=4.5)
    ax1.tick_params(color='k',  labelcolor='k', axis='both', which='both', labelsize=5, bottom=False, top=False, labelbottom=True,left=False, right=False, labelleft=True,  pad=0)      #to handle the visibility of ticks & distanc btw ticks and frame
    ax1.set_frame_on(False)     #to handle the visibility of frame



    ax1.ticklabel_format(useOffset=False, style='plain')
    fig.autofmt_xdate()    #to fix the issue of overlapping x-axis labels
    
    #Labels:
    smallgreen_circle = mlines.Line2D([], [], color='limegreen', marker='o', linestyle='None',mec="black", mew=0.3, markersize=3, label="{} path + {} start trees:\n{}-trees between each pair".format(all_path,N, n_path))
    green_triangle = mlines.Line2D([], [], color='limegreen', marker='^', linestyle='None',mec="black", mew=0.4, markersize=8, label="{} starting trees".format(len(StartTrees)))
    smallpink_circle = mlines.Line2D([], [], color='hotpink', marker='o', linestyle='None', mec="black", mew=0.25, markersize=3, label="{} trees with highest likelihood ".format(len(bestlike)))
    bigpink_circle = mlines.Line2D([], [], color='hotpink', marker='o', linestyle='None',mec="black", mew=0.4, markersize=6, label="{} best optimized tree \n of different topologies".format(N_topo))
    
    
    if paup_tree:
        bigred_circle = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
        bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_circle, bigwhite_circle])

    elif paup_MAP:
        bigred_circle = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-3]))
        bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        bigblack_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='black', mew=0.9, markersize=11, label="MAP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
        plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_circle, bigwhite_circle, bigblack_circle])
    
    else:
        bigred_circle = mlines.Line2D([], [], color='red', marker='o', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        plt.legend(bbox_to_anchor=(1.05, 0.85),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_circle])
    plt.setp(ax1.spines.values(), color='gray')


    #=============================================================================

    ax2 = plt.subplot2grid((40,345), (0,145), rowspan=50, colspan=200, projection='3d')

    ax2.plot_wireframe(XX, YY, ZZ,0.1, cmap='viridis')
    surf = ax2.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, edgecolor='none',alpha=0.25, cmap='viridis',vmin=np.nanmin(ZZ), vmax=np.nanmax(ZZ))
    
    min_value, max_value = ax2.get_zlim()       #To hadle the distance between surface and contour
    
    cset = ax2.contourf(XX, YY, ZZ, vmin=np.nanmin(ZZ), vmax=np.nanmax(ZZ), zdir='z', offset=min_value, cmap='viridis', alpha=0.3)
#    ax2.scatter3D(X[:N,0], X[:N,1], X[:N,2], marker='^', c=Likelihood[:N] , vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.5, s=45)    #starttrees
    ax2.scatter3D(X[:N,0], X[:N,1], X[:N,2], marker='^',facecolors='none',  vmin = vmin, vmax = vmax, edgecolors="black", linewidths=0.5, s=50)    #starttrees

    if paup_tree:
        ax2.scatter3D(X[-2,0],X[-2,1],X[-2,2], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax2.scatter3D(X[-1,0],X[-1,1],X[-1,2], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        opt_points2 = ax2.scatter3D(X[-opt:-1,0],X[-opt:-1,1],X[-opt:-1,2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    elif paup_MAP:
        ax2.scatter3D(X[-3,0],X[-3,1],X[-3,2], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax2.scatter3D(X[-1,0],X[-1,1],X[-1,2], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax2.scatter3D(X[-2,0],X[-2,1],X[-2,2], marker='o',facecolors='black', edgecolors="black", linewidths=0.5, s=140)      #MAP
        opt_points2 = ax2.scatter3D(X[-opt:-2,0],X[-opt:-2,1],X[-opt:-2,2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    else:
        ax2.scatter3D(X[-1,0],X[-1,1],X[-1,2], marker='o',facecolors='red', edgecolors="black", linewidths=0.5, s=150)     #best optimized
        opt_points2 = ax2.scatter3D(X[-opt:,0],X[-opt:,1],X[-opt:,2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees

    ax2.grid(linestyle='-', linewidth='5', color='green')
    points.set_clim([np.min(Likelihood), np.max(Likelihood)])



#    ax2.set_xlim3d(min_xvalue, max_xvalue);
#    ax2.set_ylim3d(min_yvalue, max_yvalue);
#    ax2.set_zlim3d(min_zvalue, max_zvalue);

    
    ax2.set_xlabel('Coordinate 1', labelpad=1,  fontsize=8)
    ax2.set_ylabel('Coordinate 2', labelpad=2,  fontsize=8)
    ax2.set_zlabel('Coordinate 3', labelpad=8,  fontsize=8)

    ax2.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
    ax2.tick_params(axis='z',labelsize=5 , pad=4)    #to change the size of numbers on axis & space between ticks and labels
    
    ax2.ticklabel_format(useOffset=False, style='plain')   # to get rid of exponential format of numbers on all axis
    ax2.view_init(21, -44)     #change the angle of 3D plot
    
    v1 = np.linspace(np.min(Likelihood), np.max(Likelihood), 7, endpoint=True)
    cbar = fig.colorbar(points, ax = ax2, shrink = 0.32, aspect = 12, pad=0.11, ticks=v1)
    cbar.ax.set_yticklabels(["{:4.4f}".format(i) for i in v1])    #To get rid of exponention expression of cbar numbers(main)
    cbar.set_label('Log Likelihood', rotation=270 , labelpad=15, fontsize=8)
    cbar.ax.tick_params(labelsize=6)   #to change the size of numbers on cbar
    cbar.outline.set_edgecolor('none')
    plt.ticklabel_format(useOffset=False)
    
    #=============================================================================
    if paup_MAP:
        ax3 = plt.subplot2grid((40,345), (27,95), rowspan=1, colspan=45, frameon=False)
    else:
        ax3 = plt.subplot2grid((40,345), (25,95), rowspan=1, colspan=45, frameon=False)
    
    if paup_tree:     #extract paup one
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):-1]
        bounds = [bounds[0]-5]+ bounds
    elif paup_MAP:     #extract paup &MAP
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):-2]
        bounds = [bounds[0]-5]+ bounds
    else:
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):]
        bounds = [bounds[0]-5]+ bounds

    cmap = mpl.colors.ListedColormap(Colors_bar)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, spacing='uniform', orientation='horizontal')
    count=0
    for label in ax3.get_xticklabels():
        count +=1
        label.set_ha("right")
        label.set_rotation(40)

    ax3.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
    ax3.set_xticklabels(["{:4.4f}".format(i) for i in bounds])    #To get rid of exponention expression of numbers
    cb2.outline.set_edgecolor('white')   #colorbar externals edge color

    # remove the first label
#    labels = ax3.get_xticklabels()
#    len_label= len(labels)
#    labels[0] = ""
#
#    if len(labels)>10:
#        kk= int(len(labels)/10)
#        for i in range(len_label-kk,len_label-1):
#            labels[i] = ""
#            for i in range(len_label-kk):
#                if i % kk !=0:
#                    labels[i] = ""
#    ax3.set_xticklabels(labels)

    labels = ax3.get_xticklabels()
    len_label= len(labels)
#    r = int(len_label/2)
    r=8
    print(f"======>len_label = {len_label}")
    if len(labels)>10:
        v2 = np.linspace(0,len_label, r,  dtype= int, endpoint=False)
        for i in range(len_label-1):
            if i not in v2:
                labels[i] = ""
    labels[0] = ""
    ax3.set_xticklabels(labels)

    

    fig.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=-0.25, right=0.98, top=1.3, wspace=None, hspace=None)  #to adjust white spaces around figure
    plt.savefig(filename)
    plt.show()










#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~interpolation_RBF:based on likelihoods (Contour & Surface)~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def interpolate_rbf_like(it, filename, M,Likelihood, bestlike, Treelist, StartTrees, optimized_BestTrees, Topologies, NUMPATHTREES):
    n = len(Treelist)
    N = len(StartTrees)
    n_path = NUMPATHTREES-2
    if paup_tree:
        opt = len(optimized_BestTrees)+1    #just PAUP
    elif paup_MAP:
        opt = len(optimized_BestTrees)+2    #both PAUP&MAP
    else:
        opt = len(optimized_BestTrees)
    
    all_path = n-opt-N
    num=200
    
    mds = MDS_sklearn(dissimilarity='precomputed', random_state=0)
    X = mds.fit_transform(M)
#    X= MDS(M,3)
    xx = np.linspace(np.min(X),np.max(X),num)
    yy =  np.linspace(np.min(X),np.max(X),num)
    zz =  np.linspace(np.min(X),np.max(X),num)
    np.savetxt ('MDS_X', X,  fmt='%s')
    
    
    XX, YY = np.meshgrid(xx,yy )
#    rbfi = Rbf(np.real(X[:,0]), np.real(X[:,1]), Likelihood, function="cubic", smooth=0)
    rbfi = Rbf(np.real(X[:,0]), np.real(X[:,1]), Likelihood)
    values = rbfi(XX, YY)   # interpolated values(likelihods)
    
#    rbf = Rbf(np.real(X[:,0]), np.real(X[:,1]), np.real(X[:,2]))
#    ZZ = rbf(XX, YY)      # interpolated z coordinates

    N_topo = len(Topologies)
    colormap = plt.cm.RdPu
    Colors = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo)]
    
    fig = plt.figure(figsize=(12,5))
    fig.set_facecolor('white')
    plt.rcParams['grid.color'] = "white" # change color
    plt.rcParams['grid.linewidth'] = 1   # change linwidth
    plt.rcParams['grid.linestyle'] = '-'
    
    ax1 = plt.subplot2grid((40,345), (14,0),rowspan=13, colspan=80)
#    contour = ax1.contourf(XX,YY,ZZ, 10, alpha=1, vmin=np.nanmin(ZZ), vmax=np.nanmax(ZZ), cmap='viridis')
    contour = ax1.contourf(XX,YY,values, 10, alpha=1, vmin=np.nanmin(values), vmax=np.nanmax(values), cmap='viridis')
    
    
#    vmin, vmax = np.nanmin(ZZ.flatten()), np.nanmax(ZZ.flatten())
    vmin, vmax = np.nanmin(values.flatten()), np.nanmax(values.flatten())
    #    print(f"vmin = {vmin} , vmax = {vmax}")
    
    #    ax1.scatter(X[:N,0], X[:N,1], c=X[:N,2] ,alpha=1, marker='^', vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.7, s=60)     #starttrees
    ax1.scatter(X[:N,0], X[:N,1], alpha=1, marker='^',facecolors='none', vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.7, s=60)     #starttrees
    
    
    #    points = ax1.scatter(X[N:,0], X[N:,1], alpha=0.8,  marker='o', c=Likelihood[N:] , cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
    #    points = ax1.scatter(X[N:,0], X[N:,1], c=X[N:,2] , alpha=1,  marker='o', cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
    
#    points = ax1.scatter(X[N:,0], X[N:,1], c=rbfi(X[N:,0],X[N:,1]) , alpha=1,  marker='o', cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
    points = ax1.scatter(X[N:,0], X[N:,1], c=Likelihood[N:] , alpha=1,  marker='o', cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
    if paup_tree:
        ax1.scatter(X[-1,0],X[-1,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax1.scatter(X[-2,0],X[-2,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        opt_points = ax1.scatter(X[-opt:-1,0],X[-opt:-1,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    
    elif paup_MAP:
        ax1.scatter(X[-3,0],X[-3,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax1.scatter(X[-1,0],X[-1,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax1.scatter(X[-2,0],X[-2,1], marker='o', facecolors='black', edgecolors="black", linewidths=0.5, s=140)     #MAP
        opt_points = ax1.scatter(X[-opt:-2,0],X[-opt:-2,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    else:
        ax1.scatter(X[-1,0],X[-1,1], marker='o', facecolors='red', edgecolors="black", linewidths=0.5, s=200)     #best optimized
        opt_points = ax1.scatter(X[-opt:,0],X[-opt:,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    
    for j, (top, c) in enumerate(zip(Topologies, Colors)):     #topologies trees
        ax1.scatter(X[top,0], X[top,1], marker='o', color=c, edgecolors="black", linewidths=0.15,s=8)
    
    #    min_xvalue, max_xvalue = ax1.get_xlim()       #To hadle the distance between boundary and contour
    #    min_yvalue, max_yvalue = ax1.get_ylim()
    min_xvalue, max_xvalue = np.min(X[:,0]),np.max(X[:,0])       #To hadle the distance between boundary and contour
    min_yvalue, max_yvalue = np.min(X[:,1]),np.max(X[:,1])
#    min_zvalue, max_zvalue = np.min(X[:,2]),np.max(X[:,2])
    x_dist =(max_xvalue-min_xvalue)/30
    y_dist =(max_yvalue-min_yvalue)/30
    
    ax1.grid(linestyle=':', linewidth='0.2', color='white', alpha=0.6)
    #    ax1.set_xlim(min_xvalue, max_xvalue)
    #    ax1.set_ylim(min_yvalue, max_yvalue)
    ax1.set_xlim(min_xvalue-x_dist, max_xvalue+x_dist)
    ax1.set_ylim(min_yvalue-y_dist, max_yvalue+y_dist)
    
    ax1.set_xlabel('Coordinate 1', labelpad=7, fontsize=8)
    ax1.set_ylabel('Coordinate 2', labelpad=3, fontsize=8)
    #    ax1.tick_params(labelsize=4.5)
    ax1.tick_params(color='k',  labelcolor='k', axis='both', which='both', labelsize=5, bottom=False, top=False, labelbottom=True,left=False, right=False, labelleft=True,  pad=0)      #to handle the visibility of ticks & distanc btw ticks and frame
    ax1.set_frame_on(False)     #to handle the visibility of frame
    
    
    
    ax1.ticklabel_format(useOffset=False, style='plain')
    fig.autofmt_xdate()    #to fix the issue of overlapping x-axis labels
    
    #Labels:
    smallgreen_circle = mlines.Line2D([], [], color='limegreen', marker='o', linestyle='None',mec="black", mew=0.3, markersize=3, label="{} path + {} start trees:\n{}-trees between each pair".format(all_path,N, n_path))
    green_triangle = mlines.Line2D([], [], color='limegreen', marker='^', linestyle='None',mec="black", mew=0.4, markersize=8, label="{} starting trees".format(len(StartTrees)))
    smallpink_circle = mlines.Line2D([], [], color='hotpink', marker='o', linestyle='None', mec="black", mew=0.25, markersize=3, label="{} trees with highest likelihood ".format(len(bestlike)))
    bigpink_circle = mlines.Line2D([], [], color='hotpink', marker='o', linestyle='None',mec="black", mew=0.4, markersize=6, label="{} best optimized tree \n of different topologies".format(N_topo))
    
    
    if paup_tree:
        bigred_circle = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
        bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_circle, bigwhite_circle])
    
    elif paup_MAP:
        bigred_circle = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-3]))
        bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        bigblack_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='black', mew=0.9, markersize=11, label="MAP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
        plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_circle, bigwhite_circle, bigblack_circle])
    
    else:
        bigred_circle = mlines.Line2D([], [], color='red', marker='o', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        plt.legend(bbox_to_anchor=(1.05, 0.85),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_circle])
    plt.setp(ax1.spines.values(), color='gray')


    #=============================================================================

    ax2 = plt.subplot2grid((40,345), (0,145), rowspan=50, colspan=200, projection='3d')
    
#    ax2.plot_wireframe(XX, YY, ZZ,0.1, cmap='viridis')
    ax2.plot_wireframe(XX, YY, values,0.1, cmap='viridis')

#    surf = ax2.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, edgecolor='none',alpha=0.25, cmap='viridis',vmin=np.nanmin(ZZ), vmax=np.nanmax(ZZ))
    surf = ax2.plot_surface(XX, YY, values, rstride=1, cstride=1, edgecolor='none',alpha=0.25, cmap='viridis',vmin=np.nanmin(values), vmax=np.nanmax(values))

    min_value, max_value = ax2.get_zlim()       #To hadle the distance between surface and contour
    
#    cset = ax2.contourf(XX, YY, ZZ, vmin=np.nanmin(ZZ), vmax=np.nanmax(ZZ), zdir='z', offset=min_value, cmap='viridis', alpha=0.3)
    cset = ax2.contourf(XX, YY, values, vmin=np.nanmin(values), vmax=np.nanmax(values), zdir='z', offset=min_value, cmap='viridis', alpha=0.3)

    #    ax2.scatter3D(X[:N,0], X[:N,1], X[:N,2], marker='^', c=Likelihood[:N] , vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.5, s=45)    #starttrees
#    ax2.scatter3D(X[:N,0], X[:N,1], X[:N,2], marker='^',facecolors='none',  vmin = vmin, vmax = vmax, edgecolors="black", linewidths=0.5, s=50)    #starttrees
    ax2.scatter3D(X[:N,0], X[:N,1], Likelihood[:N], marker='^',facecolors='none',  vmin = vmin, vmax = vmax, edgecolors="black", linewidths=0.5, s=50)    #starttrees

    if paup_tree:
#        ax2.scatter3D(X[-2,0],X[-2,1],X[-2,2], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
#        ax2.scatter3D(X[-1,0],X[-1,1],X[-1,2], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
#        opt_points2 = ax2.scatter3D(X[-opt:-1,0],X[-opt:-1,1],X[-opt:-1,2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        ax2.scatter3D(X[-2,0],X[-2,1],Likelihood[-2], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        opt_points2 = ax2.scatter3D(X[-opt:-1,0],X[-opt:-1,1],Likelihood[-opt:-1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    elif paup_MAP:
#        ax2.scatter3D(X[-3,0],X[-3,1],X[-3,2], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
#        ax2.scatter3D(X[-1,0],X[-1,1],X[-1,2], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
#        ax2.scatter3D(X[-2,0],X[-2,1],X[-2,2], marker='o',facecolors='black', edgecolors="black", linewidths=0.5, s=140)      #MAP
#        opt_points2 = ax2.scatter3D(X[-opt:-2,0],X[-opt:-2,1],X[-opt:-2,2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        ax2.scatter3D(X[-3,0],X[-3,1],Likelihood[-3], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax2.scatter3D(X[-2,0],X[-2,1],Likelihood[-2], marker='o',facecolors='black', edgecolors="black", linewidths=0.5, s=140)      #MAP
        opt_points2 = ax2.scatter3D(X[-opt:-2,0],X[-opt:-2,1],Likelihood[-opt:-2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    else:
#        ax2.scatter3D(X[-1,0],X[-1,1],X[-1,2], marker='o',facecolors='red', edgecolors="black", linewidths=0.5, s=150)     #best optimized
#        opt_points2 = ax2.scatter3D(X[-opt:,0],X[-opt:,1],X[-opt:,2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='red', edgecolors="black", linewidths=0.5, s=150)     #best optimized
        opt_points2 = ax2.scatter3D(X[-opt:,0],X[-opt:,1],Likelihood[-opt:], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees

    ax2.grid(linestyle='-', linewidth='5', color='green')
    points.set_clim([np.min(Likelihood), np.max(Likelihood)])



#    ax2.set_xlim3d(min_xvalue, max_xvalue);
#    ax2.set_ylim3d(min_yvalue, max_yvalue);
#    ax2.set_zlim3d(min_zvalue, max_zvalue);


    ax2.set_xlabel('Coordinate 1', labelpad=1,  fontsize=8)
    ax2.set_ylabel('Coordinate 2', labelpad=2,  fontsize=8)
    ax2.set_zlabel('Coordinate 3', labelpad=8,  fontsize=8)
    
    ax2.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
    ax2.tick_params(axis='z',labelsize=5 , pad=4)    #to change the size of numbers on axis & space between ticks and labels
    
    ax2.ticklabel_format(useOffset=False, style='plain')   # to get rid of exponential format of numbers on all axis
    ax2.view_init(21, -44)     #change the angle of 3D plot
    
    v1 = np.linspace(np.min(Likelihood), np.max(Likelihood), 7, endpoint=True)
    cbar = fig.colorbar(points, ax = ax2, shrink = 0.32, aspect = 12, pad=0.11, ticks=v1)
    cbar.ax.set_yticklabels(["{:4.4f}".format(i) for i in v1])    #To get rid of exponention expression of cbar numbers(main)
    cbar.set_label('Log Likelihood', rotation=270 , labelpad=15, fontsize=8)
    cbar.ax.tick_params(labelsize=6)   #to change the size of numbers on cbar
    cbar.outline.set_edgecolor('none')
    plt.ticklabel_format(useOffset=False)
    
    #=============================================================================
    if paup_MAP:
        ax3 = plt.subplot2grid((40,345), (27,95), rowspan=1, colspan=45, frameon=False)
    else:
        ax3 = plt.subplot2grid((40,345), (25,95), rowspan=1, colspan=45, frameon=False)

    if paup_tree:     #extract paup one
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):-1]
        bounds = [bounds[0]-5]+ bounds
    elif paup_MAP:     #extract paup &MAP
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):-2]
        bounds = [bounds[0]-5]+ bounds
    else:
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):]
        bounds = [bounds[0]-5]+ bounds
    
    cmap = mpl.colors.ListedColormap(Colors_bar)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, spacing='uniform', orientation='horizontal')
    count=0
    for label in ax3.get_xticklabels():
        count +=1
        label.set_ha("right")
        label.set_rotation(40)

    ax3.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
    ax3.set_xticklabels(["{:4.4f}".format(i) for i in bounds])    #To get rid of exponention expression of numbers
    cb2.outline.set_edgecolor('white')   #colorbar externals edge color

# remove the first label
#    labels = ax3.get_xticklabels()
#    len_label= len(labels)
#    labels[0] = ""
#
#    if len(labels)>10:
#        kk= int(len(labels)/10)
#        for i in range(len_label-kk,len_label-1):
#            labels[i] = ""
#            for i in range(len_label-kk):
#                if i % kk !=0:
#                    labels[i] = ""
#    ax3.set_xticklabels(labels)

    labels = ax3.get_xticklabels()
    len_label= len(labels)
#    r = int(len_label/2)
    r=8
    print(f"======>len_label = {len_label}")
    if len(labels)>10:
        v2 = np.linspace(0,len_label, r,  dtype= int, endpoint=False)
        for i in range(len_label-1):
            if i not in v2:
                labels[i] = ""
    labels[0] = ""
    ax3.set_xticklabels(labels)


    
    fig.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=-0.25, right=0.98, top=1.3, wspace=None, hspace=None)  #to adjust white spaces around figure
    plt.savefig(filename)
    plt.show()









#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~interpolation_bisplrep:based on likelihoods (Contour & Surface)~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def interpolate_rbf_bispl(it, filename, M,Likelihood, bestlike, Treelist, StartTrees, optimized_BestTrees, Topologies, NUMPATHTREES):
    n = len(Treelist)
    N = len(StartTrees)
    n_path = NUMPATHTREES-2
    if paup_tree:
        opt = len(optimized_BestTrees)+1    #just PAUP
    elif paup_MAP:
        opt = len(optimized_BestTrees)+2    #both PAUP&MAP
    else:
        opt = len(optimized_BestTrees)
    
    all_path = n-opt-N
    num=200
    
    
    #    X= MDS(M,3)
    mds = MDS_sklearn(dissimilarity='precomputed', random_state=0)
    X = mds.fit_transform(M)
    
   
    
    
    xx = np.linspace(np.min(X),np.max(X),num)
    yy =  np.linspace(np.min(X),np.max(X),num)
    zz =  np.linspace(np.min(X),np.max(X),num)
    np.savetxt ('MDS_X', X,  fmt='%s')
    
    
#    XX, YY = np.meshgrid(xx,yy )
#    rbfi = Rbf(np.real(X[:,0]), np.real(X[:,1]), Likelihood, function="cubic", smooth=0)
#    rbfi = Rbf(np.real(X[:,0]), np.real(X[:,1]), Likelihood)
#    values = rbfi(XX, YY)   # interpolated values(likelihods)

    tck = interpolate.bisplrep(np.real(X[:,0]), np.real(X[:,1]), Likelihood, s=0)
    XX, YY = np.mgrid[np.min(X):np.max(X):100j, np.min(X):np.max(X):100j]
    values = interpolate.bisplev(XX[:,0], YY[0,:], tck)       #???
    
    plt.pcolor(XX, YY,values)
    plt.show()
    
    N_topo = len(Topologies)
    colormap = plt.cm.RdPu
    Colors = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo)]
    
    fig = plt.figure(figsize=(12,5))
    fig.set_facecolor('white')
    plt.rcParams['grid.color'] = "white" # change color
    plt.rcParams['grid.linewidth'] = 1   # change linwidth
    plt.rcParams['grid.linestyle'] = '-'
    
    ax1 = plt.subplot2grid((40,345), (14,0),rowspan=13, colspan=80)
    #    contour = ax1.contourf(XX,YY,ZZ, 10, alpha=1, vmin=np.nanmin(ZZ), vmax=np.nanmax(ZZ), cmap='viridis')
    contour = ax1.contourf(XX,YY,values, 10, alpha=1, vmin=np.nanmin(values), vmax=np.nanmax(values), cmap='viridis')
    
    
    #    vmin, vmax = np.nanmin(ZZ.flatten()), np.nanmax(ZZ.flatten())
    vmin, vmax = np.nanmin(values.flatten()), np.nanmax(values.flatten())
    #    print(f"vmin = {vmin} , vmax = {vmax}")
    
    #    ax1.scatter(X[:N,0], X[:N,1], c=X[:N,2] ,alpha=1, marker='^', vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.7, s=60)     #starttrees
    ax1.scatter(X[:N,0], X[:N,1], alpha=1, marker='^',facecolors='none', vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.7, s=60)     #starttrees
    
    
    #    points = ax1.scatter(X[N:,0], X[N:,1], alpha=0.8,  marker='o', c=Likelihood[N:] , cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
    #    points = ax1.scatter(X[N:,0], X[N:,1], c=X[N:,2] , alpha=1,  marker='o', cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
    
    #    points = ax1.scatter(X[N:,0], X[N:,1], c=rbfi(X[N:,0],X[N:,1]) , alpha=1,  marker='o', cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
    points = ax1.scatter(X[N:,0], X[N:,1], c=Likelihood[N:] , alpha=1,  marker='o', cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #PathTrees
    if paup_tree:
        ax1.scatter(X[-1,0],X[-1,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax1.scatter(X[-2,0],X[-2,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        opt_points = ax1.scatter(X[-opt:-1,0],X[-opt:-1,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    
    elif paup_MAP:
        ax1.scatter(X[-3,0],X[-3,1], marker='D', facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax1.scatter(X[-1,0],X[-1,1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax1.scatter(X[-2,0],X[-2,1], marker='o', facecolors='black', edgecolors="black", linewidths=0.5, s=140)     #MAP
        opt_points = ax1.scatter(X[-opt:-2,0],X[-opt:-2,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    else:
        ax1.scatter(X[-1,0],X[-1,1], marker='o', facecolors='red', edgecolors="black", linewidths=0.5, s=200)     #best optimized
        opt_points = ax1.scatter(X[-opt:,0],X[-opt:,1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    
    for j, (top, c) in enumerate(zip(Topologies, Colors)):     #topologies trees
        ax1.scatter(X[top,0], X[top,1], marker='o', color=c, edgecolors="black", linewidths=0.15,s=8)
    
    #    min_xvalue, max_xvalue = ax1.get_xlim()       #To hadle the distance between boundary and contour
    #    min_yvalue, max_yvalue = ax1.get_ylim()
    min_xvalue, max_xvalue = np.min(X[:,0]),np.max(X[:,0])       #To hadle the distance between boundary and contour
    min_yvalue, max_yvalue = np.min(X[:,1]),np.max(X[:,1])
    #    min_zvalue, max_zvalue = np.min(X[:,2]),np.max(X[:,2])
    x_dist =(max_xvalue-min_xvalue)/30
    y_dist =(max_yvalue-min_yvalue)/30

    ax1.grid(linestyle=':', linewidth='0.2', color='white', alpha=0.6)
    #    ax1.set_xlim(min_xvalue, max_xvalue)
    #    ax1.set_ylim(min_yvalue, max_yvalue)
    ax1.set_xlim(min_xvalue-x_dist, max_xvalue+x_dist)
    ax1.set_ylim(min_yvalue-y_dist, max_yvalue+y_dist)
    
    ax1.set_xlabel('Coordinate 1', labelpad=7, fontsize=8)
    ax1.set_ylabel('Coordinate 2', labelpad=3, fontsize=8)
    #    ax1.tick_params(labelsize=4.5)
    ax1.tick_params(color='k',  labelcolor='k', axis='both', which='both', labelsize=5, bottom=False, top=False, labelbottom=True,left=False, right=False, labelleft=True,  pad=0)      #to handle the visibility of ticks & distanc btw ticks and frame
    ax1.set_frame_on(False)     #to handle the visibility of frame
    
    
    
    ax1.ticklabel_format(useOffset=False, style='plain')
    fig.autofmt_xdate()    #to fix the issue of overlapping x-axis labels
    
    #Labels:
    smallgreen_circle = mlines.Line2D([], [], color='limegreen', marker='o', linestyle='None',mec="black", mew=0.3, markersize=3, label="{} path + {} start trees:\n{}-trees between each pair".format(all_path,N, n_path))
    green_triangle = mlines.Line2D([], [], color='limegreen', marker='^', linestyle='None',mec="black", mew=0.4, markersize=8, label="{} starting trees".format(len(StartTrees)))
    smallpink_circle = mlines.Line2D([], [], color='hotpink', marker='o', linestyle='None', mec="black", mew=0.25, markersize=3, label="{} trees with highest likelihood ".format(len(bestlike)))
    bigpink_circle = mlines.Line2D([], [], color='hotpink', marker='o', linestyle='None',mec="black", mew=0.4, markersize=6, label="{} best optimized tree \n of different topologies".format(N_topo))
    
    
    if paup_tree:
        bigred_circle = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
        bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_circle, bigwhite_circle])

    elif paup_MAP:
        bigred_circle = mlines.Line2D([], [], color='red', marker='D', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-3]))
        bigwhite_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='white', mew=0.9, markersize=11, label="PAUP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        bigblack_circle = mlines.Line2D([], [],  marker='o', linestyle='None',mec="black",mfc='black', mew=0.9, markersize=11, label="MAP best optimized tree \nLogLike {:4.4f}".format(Likelihood[-2]))
        plt.legend(bbox_to_anchor=(1.05, 0.95),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_circle, bigwhite_circle, bigblack_circle])

    else:
        bigred_circle = mlines.Line2D([], [], color='red', marker='o', linestyle='None',mec="black", mew=0.7, markersize=11, label="PATHTREES best optimized tree \nLogLike {:4.4f}".format(Likelihood[-1]))
        plt.legend(bbox_to_anchor=(1.05, 0.85),  fontsize='x-small', labelspacing = 1.5, frameon=False, scatterpoints=1, handles=[ smallgreen_circle, green_triangle, smallpink_circle, bigpink_circle,   bigred_circle])
    plt.setp(ax1.spines.values(), color='gray')


#=============================================================================

    ax2 = plt.subplot2grid((40,345), (0,145), rowspan=50, colspan=200, projection='3d')

    #    ax2.plot_wireframe(XX, YY, ZZ,0.1, cmap='viridis')
    ax2.plot_wireframe(XX, YY, values,0.1, cmap='viridis')
    
    #    surf = ax2.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, edgecolor='none',alpha=0.25, cmap='viridis',vmin=np.nanmin(ZZ), vmax=np.nanmax(ZZ))
    surf = ax2.plot_surface(XX, YY, values, rstride=1, cstride=1, edgecolor='none',alpha=0.25, cmap='viridis',vmin=np.nanmin(values), vmax=np.nanmax(values))
    
    min_value, max_value = ax2.get_zlim()       #To hadle the distance between surface and contour
    
    #    cset = ax2.contourf(XX, YY, ZZ, vmin=np.nanmin(ZZ), vmax=np.nanmax(ZZ), zdir='z', offset=min_value, cmap='viridis', alpha=0.3)
    cset = ax2.contourf(XX, YY, values, vmin=np.nanmin(values), vmax=np.nanmax(values), zdir='z', offset=min_value, cmap='viridis', alpha=0.3)
    
    #    ax2.scatter3D(X[:N,0], X[:N,1], X[:N,2], marker='^', c=Likelihood[:N] , vmin = vmin, vmax = vmax, cmap='viridis',  edgecolors="black", linewidths=0.5, s=45)    #starttrees
    #    ax2.scatter3D(X[:N,0], X[:N,1], X[:N,2], marker='^',facecolors='none',  vmin = vmin, vmax = vmax, edgecolors="black", linewidths=0.5, s=50)    #starttrees
    ax2.scatter3D(X[:N,0], X[:N,1], Likelihood[:N], marker='^',facecolors='none',  vmin = vmin, vmax = vmax, edgecolors="black", linewidths=0.5, s=50)    #starttrees
    
    if paup_tree:
        #        ax2.scatter3D(X[-2,0],X[-2,1],X[-2,2], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        #        ax2.scatter3D(X[-1,0],X[-1,1],X[-1,2], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        #        opt_points2 = ax2.scatter3D(X[-opt:-1,0],X[-opt:-1,1],X[-opt:-1,2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        ax2.scatter3D(X[-2,0],X[-2,1],Likelihood[-2], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        opt_points2 = ax2.scatter3D(X[-opt:-1,0],X[-opt:-1,1],Likelihood[-opt:-1], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    elif paup_MAP:
        #        ax2.scatter3D(X[-3,0],X[-3,1],X[-3,2], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        #        ax2.scatter3D(X[-1,0],X[-1,1],X[-1,2], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        #        ax2.scatter3D(X[-2,0],X[-2,1],X[-2,2], marker='o',facecolors='black', edgecolors="black", linewidths=0.5, s=140)      #MAP
        #        opt_points2 = ax2.scatter3D(X[-opt:-2,0],X[-opt:-2,1],X[-opt:-2,2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        ax2.scatter3D(X[-3,0],X[-3,1],Likelihood[-3], marker='D',facecolors='red', edgecolors="black", linewidths=0.5, s=220)     #best optimized
        ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='white', edgecolors="black", linewidths=0.5, s=140)      #PAUP
        ax2.scatter3D(X[-2,0],X[-2,1],Likelihood[-2], marker='o',facecolors='black', edgecolors="black", linewidths=0.5, s=140)      #MAP
        opt_points2 = ax2.scatter3D(X[-opt:-2,0],X[-opt:-2,1],Likelihood[-opt:-2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
    else:
        #        ax2.scatter3D(X[-1,0],X[-1,1],X[-1,2], marker='o',facecolors='red', edgecolors="black", linewidths=0.5, s=150)     #best optimized
        #        opt_points2 = ax2.scatter3D(X[-opt:,0],X[-opt:,1],X[-opt:,2], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        ax2.scatter3D(X[-1,0],X[-1,1],Likelihood[-1], marker='o',facecolors='red', edgecolors="black", linewidths=0.5, s=150)     #best optimized
        opt_points2 = ax2.scatter3D(X[-opt:,0],X[-opt:,1],Likelihood[-opt:], marker='o',c=Colors, edgecolors="black", linewidths=0.5, s=40)      #optimized trees
        
        ax2.grid(linestyle='-', linewidth='5', color='green')
        points.set_clim([np.min(Likelihood), np.max(Likelihood)])

    
    
    #    ax2.set_xlim3d(min_xvalue, max_xvalue);
    #    ax2.set_ylim3d(min_yvalue, max_yvalue);
    #    ax2.set_zlim3d(min_zvalue, max_zvalue);
    
    
    ax2.set_xlabel('Coordinate 1', labelpad=1,  fontsize=8)
    ax2.set_ylabel('Coordinate 2', labelpad=2,  fontsize=8)
    ax2.set_zlabel('Coordinate 3', labelpad=8,  fontsize=8)
    
    ax2.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
    ax2.tick_params(axis='z',labelsize=5 , pad=4)    #to change the size of numbers on axis & space between ticks and labels
    
    ax2.ticklabel_format(useOffset=False, style='plain')   # to get rid of exponential format of numbers on all axis
    ax2.view_init(21, -44)     #change the angle of 3D plot
    
    v1 = np.linspace(np.min(Likelihood), np.max(Likelihood), 7, endpoint=True)
    cbar = fig.colorbar(points, ax = ax2, shrink = 0.32, aspect = 12, pad=0.11, ticks=v1)
    cbar.ax.set_yticklabels(["{:4.4f}".format(i) for i in v1])    #To get rid of exponention expression of cbar numbers(main)
    cbar.set_label('Log Likelihood', rotation=270 , labelpad=15, fontsize=8)
    cbar.ax.tick_params(labelsize=6)   #to change the size of numbers on cbar
    cbar.outline.set_edgecolor('none')
    plt.ticklabel_format(useOffset=False)
    
    #=============================================================================
    if paup_MAP:
        ax3 = plt.subplot2grid((40,345), (27,95), rowspan=1, colspan=45, frameon=False)
    else:
        ax3 = plt.subplot2grid((40,345), (25,95), rowspan=1, colspan=45, frameon=False)
    
    if paup_tree:     #extract paup one
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):-1]
        bounds = [bounds[0]-5]+ bounds
    elif paup_MAP:     #extract paup &MAP
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):-2]
        bounds = [bounds[0]-5]+ bounds
    else:
        Colors_bar = [colormap(i) for i in np.linspace(0.1, 0.9,N_topo+1)]
        bounds= Likelihood[-(opt):]
        bounds = [bounds[0]-5]+ bounds
    
    cmap = mpl.colors.ListedColormap(Colors_bar)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, spacing='uniform', orientation='horizontal')
    count=0
    for label in ax3.get_xticklabels():
        count +=1
        label.set_ha("right")
        label.set_rotation(40)

    ax3.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
    ax3.set_xticklabels(["{:4.4f}".format(i) for i in bounds])    #To get rid of exponention expression of numbers
    cb2.outline.set_edgecolor('white')   #colorbar externals edge color

# remove the first label
#    labels = ax3.get_xticklabels()
#    len_label= len(labels)
#    labels[0] = ""
#
#    if len(labels)>10:
#        kk= int(len(labels)/10)
#        for i in range(len_label-kk,len_label-1):
#            labels[i] = ""
#            for i in range(len_label-kk):
#                if i % kk !=0:
#                    labels[i] = ""
#    ax3.set_xticklabels(labels)

    labels = ax3.get_xticklabels()
    len_label= len(labels)
    #    r = int(len_label/2)
    r=8
    print(f"======>len_label = {len_label}")
    if len(labels)>10:
        v2 = np.linspace(0,len_label, r,  dtype= int, endpoint=False)
        for i in range(len_label-1):
            if i not in v2:
                labels[i] = ""
    labels[0] = ""
    ax3.set_xticklabels(labels)



    fig.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=-0.25, right=0.98, top=1.3, wspace=None, hspace=None)  #to adjust white spaces around figure
    plt.savefig(filename)
    plt.show()


def voronoi2d(points,Z, ax1):
    minx = np.min([i[0] for i in points])
    miny = np.min([i[1] for i in points])
    maxx = np.max([i[0] for i in points])
    maxy = np.max([i[1] for i in points])
    print(minx,miny,maxx,maxy)
    points = np.array(points.tolist() + [[-999,0], [999,0], [0,-999], [0,999]])
    print(points)
    h=999999
    # find min/max values for normalization
    minima = min(Z)
    maxima = max(Z)
    Z.extend([-h,-h,-h,-h])
    Z = np.array(Z)
    #speed = [x-minima+1 for x in speed]
    #maxima = maxima - minima + 1
    #minima = 1
    # generate Voronoi tessellation
    vor = Voronoi(points)
    hi = np.amax(Z)
    # normalize chosen colormap
    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    #norm = mpl.colors.LogNorm(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)    
    # plot Voronoi diagram, and fill finite regions with color mapped from speed value
    fig = voronoi_plot_2d(vor, ax1, show_points=True, show_vertices=False, s=1,line_width=0.1)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(Z[r]))
    print("Voronoi finished")
    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    boundary_convexhull    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def boundary_convexhull(M,Likelihood,treelist):
    meth= 'cubic'
    num=100
    X= MDS(M,2)
    xx = np.linspace(np.min(X),np.max(X),num)
    yy =  np.linspace(np.min(X),np.max(X),num)
    XX, YY = np.meshgrid(xx,yy)          #We DONT know the likelihood values of these points
    values = griddata((np.real(X[:,0]), np.real(X[:,1])), Likelihood, (XX, YY), method=meth)
    
    fig = plt.figure(figsize=(12,5))
    
    ax1 = plt.subplot2grid((20,40), (3,0),rowspan=14, colspan=12)
    contour = ax1.contourf(XX, YY, values, alpha=0.9, vmin=np.nanmin(values), vmax=np.nanmax(values))
    points=ax1.scatter(X[:,0], X[:,1], alpha=0.8,  marker='o', c=Likelihood , cmap='viridis',  edgecolors="black", linewidths=0.3, s=5)  #Trees
    
    ax1.grid(linestyle=':', linewidth='0.1', color='grey', alpha=0.6)
    ax1.set_xlim(np.min(X[:,0])-0.01, np.max(X[:,0])+0.01)
    ax1.set_ylim(np.min(X[:,1])-0.01, np.max(X[:,1])+0.01)
    ax1.set_xlabel('Coordinate 1', labelpad=3, fontsize=8)
    ax1.set_ylabel('Coordinate 2', labelpad=3, fontsize=8)
    ax1.tick_params(labelsize=4.5)
    ax1.ticklabel_format(useOffset=False, style='plain')
    
    
    hull = ConvexHull(X)
    # Get the indices of the hull points.
    hull_indices = hull.vertices
    print("Boundary trees idx =",hull_indices)
    print("len of Boundary trees =",len(hull_indices))
    # These are the actual points.
    hull_pts = X[hull_indices, :]
    Boundary_Trees = [treelist[i] for i in hull_indices]
    Boundary_Trees = [s.replace('\n', '') for s in Boundary_Trees]
#    np.savetxt ('Boundary_Trees', Boundary_Trees,  fmt='%s')
#    print("\n\nBoundary_Trees =\n",Boundary_Trees)
    
    ax2 = plt.subplot2grid((20,40), (3,16), rowspan=14, colspan=16)
    v1 = np.linspace(np.min(Likelihood), np.max(Likelihood), 7, endpoint=True)
    cbar = fig.colorbar(points, ax = ax2, shrink = 1, aspect = 15, pad=0.1, ticks=v1)
    ax2.scatter(X[:,0], X[:,1], marker='o', c=Likelihood,  s=5)
    ax2.scatter(hull_pts[:,0], hull_pts[:,1], marker='o', c='r',  s=40)
    plt.fill(hull_pts[:,0], hull_pts[:,1], fill=False, facecolor='none', edgecolor='r',  linewidth=1, ls='--')
    cbar.set_label('Log Likelihood', rotation=270 , labelpad=15, fontsize=10)
    ax2.set_xlabel('Coordinate 1', labelpad=3,  fontsize=8)
    ax2.set_ylabel('Coordinate 2', labelpad=3,  fontsize=8)
    ax2.set_xlim(np.min(X[:,0])-0.01, np.max(X[:,0])+0.01)
    ax2.set_ylim(np.min(X[:,1])-0.01, np.max(X[:,1])+0.01)
    ax2.tick_params(labelsize=4.5 , pad=1)    #to change the size of numbers on axis & space between ticks and labels
    ax2.ticklabel_format(useOffset=False, style='plain')   # to get rid of exponential format of numbers on all axis
    cbar.ax.tick_params(labelsize=6)   #to change the size of numbers on cbar
#    fig.autofmt_xdate()    #to fix the issue of overlapping x-axis labels
    plt.savefig('Contour_Boundary_{}.pdf'.format(meth), format='pdf')
#    plt.show()
    return Boundary_Trees







if __name__ == "__main__":
    GTPOUTPUT = 'results/output.txt'
    likelihoodfile = "results/likelihoods"
    treefile = "results/treelist"
    pathtreefile = "results/pathTrees"
    starttreefile = "results/startTrees"
    Likelihood,treelist,pathlist, StartTrees = read_from_files(likelihoodfile,treefile,pathtreefile, starttreefile)

    n = len(treelist)
    NUMBESTTREES = n
    N= len(pathlist)
    bestlike = best_likelihoods(Likelihood,NUMBESTTREES)
    distances = read_GTP_distances(n,GTPOUTPUT)
    file = "TESTPLOT.pdf"
    plot_MDS(file,N,n,distances, Likelihood, bestlike, treelist, pathlist)
    interpolate_grid(distances,Likelihood, bestlike, StartTrees)
