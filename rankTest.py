import numpy as np
import scipy.linalg as li
import scipy.sparse as sp
import scipy.sparse.linalg as spli
from convectionDiffusion import *
from multifrontal import *
import cProfile
import pstats
from hodlrDirectSolver import *
import gc

def counterFunc(r):
    global iterctr
    iterctr=iterctr+1

ns = [800];
#dx = 0.001
dts = [0.001, 0.01, 0.1, 1];
Ds = [0.001, 0.01, 0.1, 1];
approxTol = [1e-3, 1e-6, 1e-8];
Vss = [1, 51, 100]
minNodeSize = 10

gmresTol = 1e-6

f = open('C:\\Stanford\\Projects\\Darve\\RankTestResults\\CircularVelocityField9.txt', 'a');
f.write('n\tdx\tdt\tD\tVs\ttol\terror\tranks\tgmresTol\tgmresFlag\tgmresIter\n')

for n in ns:
    X,Y = np.meshgrid(xrange(0,n+2),xrange(0,n+2))
    dx = n**-1
    for Vs in Vss:
        vx = -Vs*(Y-(n+2)/2 * np.ones(n+2))/n
        vy = Vs*(X-(n+2)/2 * np.ones(n+2))/n
        b = np.random.rand(n,1)
        for dt in dts:
            for D in Ds:
                Dmtx = D*np.ones((n+2,n+2));
                A, p, levels = ConstructConvectionDiffusionMatrix(Dmtx, vx, vy, dt, dx, n, minNodeSize);
                for tol in approxTol:
                    try:
                        etree = supernodalMultifrontal(A, np.reshape(p, (n,n), 'F'), levels, minNodeSize, 1e-12)
                        frontalMatrix = etree.getNode(etree.numberOfSupernodes-1).A_matrix
                        x, ranks = HODLRDirectSolverSMWReturnRanks(frontalMatrix, b, n/10, tol)
                        x_true = li.solve(frontalMatrix, b)
                        error = li.norm(np.reshape(x_true,(n,1),'F')-x)
        
                        iterctr = 0
                        x, flag = spli.gmres(frontalMatrix, b, tol=gmresTol, maxiter=1000, callback=counterFunc)                 
                        
                        text = '%d\t%1.4f\t%1.4f\t%1.4f\t%6.3f\t%1.2e\t%1.3e\t%1.3e\t%d\t%d\t['%(n, dx, dt, D, Vs, tol, error, gmresTol, flag, iterctr)
                        f.write(text)
                        for rank in ranks:
                            f.write("%d "%rank)
                        f.write(']\n')
                        f.flush()
                        print n, dx, dt, D, tol
                    except:
                        None
                    

f.close()