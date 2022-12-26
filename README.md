<div align="center"><img src="https://raw.githubusercontent.com/TaraKhodaei/pathtrees_project/main/images/pathtrees_logo.jpg" width="420"/></div>

# PATHTREES
Python package **PATHTREES** enables the construction, visualization and exploration of the continuous tree landscape interior of the convex hull of given starting trees, using insights from the Billera-Holmes-Vogtmann treespace.


## Usage

    pathtrees.py [-h] [-o OUTPUTDIR] [-v] [-p PLOTFILE] [-n NUMPATHTREES]
                        [-b NUMBESTTREES] [-r NUM_RANDOM_TREES] [-g OUTGROUP]
                        [-i NUM_ITERATIONS] [-e] [-hull] [-gtp] [-nel] [-c COMPARE_TREES]
                        [-interp INTERPOLATION]
                        STARTTREES DATAFILE



## Positional Arguments

    STARTTREES     mandatory input file that holds a set of trees in Newick format

    DATAFILE       mandatory input file that holds a sequence data set in PHYLIP format


## Optional Arguments

optional arguments:
    
    -h, --help            
                    show this help message and exit
    
    -o OUTPUTDIR, --output OUTPUTDI            
                    directory that holds the output files
                        
    -v, --verbose            
                    Do not remove the intermediate files generated by GPT
    
    -p PLOTFILE, --plot PLOTFILE
                    Create an MDS plot from the generated distances
                        
    -n NUMPATHTREES, --np NUMPATHTREES, --numpathtrees NUMPATHTREES
                    Number of trees along the path between two initial trees
                        
    -b NUMBESTTREES, --best NUMBESTTREES, --numbesttrees NUMBESTTREES
                    Number of trees selected from the best likliehood trees for the next round of refinement
                        
    -r NUM_RANDOM_TREES, --randomtrees NUM_RANDOM_TREES
                    Generate num_random_trees rooted trees using the sequence data individual names.
                        
    -g OUTGROUP, --outgroup OUTGROUP
                    Forces an outgroup when generating random trees.
                        
    -i NUM_ITERATIONS, --iterate NUM_ITERATIONS
                    Takes the trees, generates the pathtrees, then picks the 10 best trees and reruns pathtrees,                     this will add an iteration number to the outputdir, and also adds iteration to the plotting.
                        
    -e, --extended        
                    If the phylip dataset is in the extended format, use this.
    
    -hull, --convex_hull
                    Extracts the convex hull of input sample trees and considers them as starting trees in the                       first iteration to generate pairwise pathtrees. If false, it directly considers input sample                     trees as starting trees in the first iteration to generate pairwise pathtrees.
                        
    -gtp, --gtp_distance    
                    Use GTP derived geodesic distance for MDS plotting [slower], if false use weighted Robinson-                     Foulds distance for MDS plotting [faster]
                        
    -nel, --neldermead
                    Use Nelder-Mead optimization method to optimize branchlengths [slower], if false use Newton-                     Raphson to optimize branchlengths [fast]
                        
    -c, --compare_trees
                    String "D1" considers the first dataset (D_1) with two trees to be compared (PAUP and MAP)                       with the best tree of PATHTREES, string "D2" considers the second dataset (D_2) with two                         trees to be compared (PAUP and RAxML) with the best tree of PATHTREES, string  "user_trees"                     considers user_trees to be compared with the best tree of PATHTREES, otherwise it considers                     nothing to be compared
                    
    -interp, --interpolate
                    Use interpolation scipy.interpolate.griddata for interpolation [more overshooting], or use                       scipy.interpolate.Rbf [less overshooting]. String "rbf" considers scipy.interpolate.Rbf,                         Radial basis function (RBF) thin-plate spline interpolation, with default smoothness=1e-10.                     String "rbf,s_value", for example "rbg,0.0001", considers scipy.interpolate.Rbf with                             smoothness= s_value= 0.0001. String "cubic" considers scipy.interpolate.griddata, cubic                         spline interpolation. Otherwise, with None interpolation, it considers default                                   scipy.interpolate.Rbf with smoothness=1e-10

## Example 1
    python pathtrees.py -n 3 -gtp -c D1 -p myplot -o output FirstData FirstData.phy

<div align="center"><img src="https://raw.githubusercontent.com/TaraKhodaei/pathtrees_project/main/images/fig5_GTP_rbf_s1e-10_n3.png" width="900"/></div>

## Example 2
    python pathtrees.py -e -i 2  -n 6,7 -b 100 -c D2 -p myplot -o output SecondData SecondData.phy

<div align="center"><img src="https://raw.githubusercontent.com/TaraKhodaei/pathtrees_project/main/images/fig7_iter1_RF_rbf_s1e-10_n6_b100.png" width="900"/></div>
<div align="center"><img src="https://raw.githubusercontent.com/TaraKhodaei/pathtrees_project/main/images/fig7_iter2_RF_rbf_s1e-10_n7.png" width="900"/></div>



```python

```
