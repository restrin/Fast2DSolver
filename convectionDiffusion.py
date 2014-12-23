import numpy as np
import scipy.linalg as li
import scipy.sparse as sp
import matplotlib.pylab as pl
import scipy.sparse.linalg as spli
import math
from scipy import signal

def ConstructConvectionDiffusionMatrix( D, vx, vy, dt, dx, n, minNodeSize ):
    """ 
    Construct the (n*n)x(n*n) Convection-Diffusion matrix to be solved
    Input:
            D   - Diffusion constant
            vx  - (n+2)x(n+2) matrix of velocity in X direction.
                    Includes boundary.
            vy  - (n+2)x(n+2) matrix of velocity in Y direction.
                    Includes boundary.
            dt  - Time discretization.
            dx  - Space discretization.
            n   - Number of nodes along one direction.
    
    Output:
            A   - matrix for problem, in CSC sparse format
            p   - permutation vector mapping column ordered nodes into
                  nested dissection.
            levels - number of supernodes in the nested dissection ordering. 
    """
    rows = np.zeros(n*n + (2*n*n-2*n) + (2*n*n-2*n));
    cols = np.zeros(n*n + (2*n*n-2*n) + (2*n*n-2*n));
    data = np.zeros(n*n + (2*n*n-2*n) + (2*n*n-2*n));
    ctr = 0;
    for i in xrange(n*n):
        x = i%n;
        y = i/n;
        rows[ctr] = i;
        cols[ctr] = i;
        data[ctr] = 1 - dt*(-4*D[x,y]/(dx*dx) \
            - ((vx[x,y+1]-vx[x,y-1] + vy[x+1,y] - vy[x-1,y])/(2*dx)));
        ctr = ctr+1;
#        A[i,i] = 1 - dt*(-4*D/(dx*dx) \
#            - ((vx[x,y+1]-vx[x,y-1] + vy[x+1,y] - vy[x-1,y])/(2*dx)));
            
        # Diffusion might be wrong
        if ((i+1)%n != 0 and i < n*n):
            rows[ctr] = i;
            cols[ctr] = i+1;
            data[ctr] = -dt*(D[x,y]/(dx*dx) - vy[x,y]/(2*dx) - (D[x,y+1]-D[x,y-1])/(4*dx*dx));
            ctr = ctr+1;
#            A[i,i+1] = -dt*(D/(dx*dx) - vy[x,y]/(2*dx));
            rows[ctr] = i+1;
            cols[ctr] = i;
            data[ctr] = -dt*(D[x,y]/(dx*dx) + vy[x,y]/(2*dx) + (D[x,y+1]-D[x,y-1])/(4*dx*dx));
            ctr = ctr+1;
#            A[i+1,i] = -dt*(D/(dx*dx) + vy[x,y]/(2*dx));
            
        if (i < n*n-n):
            rows[ctr] = i;
            cols[ctr] = i+n;
            data[ctr] = -dt*(D[x,y]/(dx*dx) - vx[x,y]/(2*dx) - (D[x+1,y]-D[x-1,y])/(4*dx*dx));
            ctr = ctr+1;
#            A[i,i+n] = -dt*(D/(dx*dx) - vx[x,y]/(2*dx));
            rows[ctr] = i+n;
            cols[ctr] = i;
            data[ctr] = -dt*(D[x,y]/(dx*dx) + vx[x,y]/(2*dx) + (D[x+1,y]-D[x-1,y])/(4*dx*dx));
            ctr = ctr+1;
#            A[i+n,i] = -dt*(D/(dx*dx) + vx[x,y]/(2*dx));
            
    A = sp.coo_matrix((data, (rows, cols)), shape=(n*n,n*n));
    p, levels = nestedDissectionPermutation(n, minNodeSize);
    symmetricPermute(A, p);
    return sp.csc_matrix(A), p, levels

def nestedDissectionPermutation(n, minNodeSize):
    """
    Determines permutation vector p to order an n-by-n mesh originally
    ordered by columns, into nested dissection ordering.
    Input:
            n   - length of mesh in one direction.
    Output:
            permutation vector p, to permute rows/cols of A into nested
            nested dissection.
            number of separators in the nested dissection
    """
    P = np.zeros((n,n));
    levels = nestedDissectionOrdering2(P, -1, 1, minNodeSize);
    p = np.reshape(P, n*n, 'F');
    p = np.argsort(p, kind='mergesort');
    p = np.argsort(p, kind='mergesort');
    return p, -levels

def nestedDissectionOrdering(P, level, direction, minNodeSize):
    if (P.shape[direction] <= minNodeSize):
        P[:,:] = level;
    else:
        n = P.shape[direction];
        if (direction == 0):
            P[n/2,:] = level;
            nestedDissectionOrdering(P[0:n/2,:], 2*level - 1, (direction+1)%2, minNodeSize);
            nestedDissectionOrdering(P[n/2+1:,:],2*level, (direction+1)%2, minNodeSize);
        else:
            P[:,n/2] = level;
            nestedDissectionOrdering(P[:,0:n/2], 2*level-1, (direction+1)%2, minNodeSize);
            nestedDissectionOrdering(P[:,n/2+1:], 2*level, (direction+1)%2, minNodeSize);   

### This one gives correct postordering automatically (I think)
def nestedDissectionOrdering2(P, level, direction, minNodeSize):
    """
        Given an empty mesh (2d-array), replaces the entries in the mesh
        with the corresponding (negative) level of the separator. Levels are
        negative so that when nodes are ordered, the top level separator comes
        last.
        
        input:
                P - empty mesh
                level - Initial input should be -1
                direction - 0 for horizontal, 1 for vertical
        output:
                Overwrites P with the (negative) levels of the node's separator.
    """
    if (P.shape[direction] <= minNodeSize):
        P[:,:] = level;
        return level
    else:
        n = P.shape[direction];
        if (direction == 0):
            P[n/2,:] = level;
            level = nestedDissectionOrdering2(P[n/2+1:,:], level - 1, (direction+1)%2, minNodeSize);
            return nestedDissectionOrdering2(P[0:n/2,:], level - 1, (direction+1)%2, minNodeSize);
        else:
            P[:,n/2] = level;
            level = nestedDissectionOrdering2(P[:,n/2+1:], level-1, (direction+1)%2, minNodeSize);
            return nestedDissectionOrdering2(P[:,0:n/2], level-1, (direction+1)%2, minNodeSize);
                           
def permuteIndices(rc, p):
    """
        Permutes entries in rc according to p.
    """
    for i in xrange(len(rc)):
         rc[i] = p[rc[i]];
    return rc

def symmetricPermute(A, p):
    """
        Permutes the entries of A symmetrically by working on the row and
        column indices.
    """
    # Assume A is COO_matrix
    permuteIndices(A.row, p)
    permuteIndices(A.col, p)   
                  
def BackwardEulerConvectionDiffusion2D(A, f, steps):
    n2 = (A.shape[0]);
    C = np.zeros((n2,steps+1));
    C[:,0] = np.reshape(f, n2, 'F');
    
    for i in xrange(steps):
        C[:,i+1] = spli.spsolve(A,C[:,i]);
        
    return C

def gauss1d(sigma):
    # Obtain correct dimensions for filter
    n = math.ceil(6*sigma)
    if n % 2 == 0: n=n+1
    # Generate 1D array of inputs
    res = np.linspace(-(n-1)/2, (n-1)/2, n)
    vexp = np.vectorize(math.exp)
    res = vexp(- res*res / (2*sigma*sigma))
    res = res/sum(res)
    return res  

def gauss2d(sigma):
    x = gauss1d(sigma)
    y = gauss1d(sigma)
    x = x[np.newaxis]
    y = np.transpose(y[np.newaxis])
    return np.outer(x, y)     
                  
def smoothRandomDiffusion(n, sigma):
    """
        Generates an (n+2)x(n+2) array of random diffusion values, smoothed
        by a Gaussian kernel.
    """
    D = np.random.rand(n+2,n+2)
    gfilter = gauss2d(sigma)
    D = signal.convolve2d(D, gfilter, 'same')
    return D

    