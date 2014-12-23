import numpy as np
import scipy.linalg as li
import scipy.sparse as sp
import scipy.sparse.linalg as spli
from eliminationTree import *
from sets import *
from math import *
import cProfile

### A is CSC sparse matrix
def symbChol(A):
    n = A.shape[1]
    etree =  EliminationTree(int(sqrt(n)))
    cols = [0 for i in range(n+1)]
    rows = []
    for i in range(n-1):
        Lj = Set(filter(lambda x: x >= i, A.indices[A.indptr[i]:A.indptr[i+1]]))
#        Lj = filter(lambda x: x >= i, A.indices[A.indptr[i]:A.indptr[i+1]])
                
        for j in etree.getNode(i).getChildren():
#            Lj = Lj + filter(lambda x: x > i, rows[cols[j]:cols[j+1]])
            Lj = Lj.union(filter(lambda x: x > i, rows[cols[j]:cols[j+1]]));

        rows = rows + sorted(list(Lj))
        cols[i+1] = len(rows)
        # Update elimination tree
        parent = min(filter(lambda x: x != i, rows[cols[i]:cols[i+1]]))
        etree.getNode(i).addParent(parent)
        etree.getNode(parent).addChild(i)
        
    rows = rows + [n-1]
    cols[-1] = len(rows)
    return sp.csc_matrix((np.ones(len(rows)), rows, cols), shape=(n,n)), etree
    
def supernodalSymbChol(A, superNodalEtree):
    n = superNodalEtree.numberOfSupernodes
    N = superNodalEtree.numberOfNodes
    cols = [0 for i in range(A.shape[0]+1)]
    rows = [0 for i in range(3*N*int(log(N,2)))] ## Need better estimate
    ctr = 0
    for i in xrange(n):
        superNode = superNodalEtree.getNode(i)
        first = superNode.nodes[0]
        last = superNode.nodes[-1]
        
        Lj = Set(filter(lambda x: x > last, A.indices[A.indptr[first]:A.indptr[last+1]]))
        
        for j in superNode.getChildren():
            c = superNodalEtree.nodes[j].nodes[-1]
#            Lj = Lj + filter(lambda x: x > i, rows[cols[j]:cols[j+1]])
            Lj = Lj.union(filter(lambda x: x > last, rows[cols[c]:cols[c+1]]));
        
        Lj = sorted(list(Lj))

#        Lj, first, last = doStuff1(superNode, A, superNodalEtree, rows, cols)

#        rows, cols = doStuff2(rows, cols, first, last, Lj)
        for k in xrange(first, last+1):
            rows[ctr:ctr+last+1-k + len(Lj)] = range(k,last+1) + Lj
            cols[k+1] = ctr + (last-k+1) + len(Lj)
            ctr = ctr + last+1-k+len(Lj)
           
    return rows[:ctr], cols #sp.csc_matrix((np.ones(ctr), rows[:ctr], cols), shape=(A.shape))       
     
        
def doStuff1(superNode, A, superNodalEtree, rows, cols):
    first = superNode.nodes[0]
    last = superNode.nodes[-1]
        
    Lj = Set(filter(lambda x: x > last, A.indices[A.indptr[first]:A.indptr[last+1]]))
        
    for j in superNode.getChildren():
        c = superNodalEtree.nodes[j].nodes[-1]
#        Lj = Lj + filter(lambda x: x > i, rows[cols[j]:cols[j+1]])
        Lj = Lj.union(filter(lambda x: x > last, rows[cols[c]:cols[c+1]]));
 
    return sorted(list(Lj)), first, last  
    
def doStuff2(rows, cols, first, last, Lj):
    rowStart = len(rows)
    for k in range(first, last+1):
        rows = rows + range(k,last+1) + Lj
        cols[k+1] = rowStart + (last-k+1) + len(Lj)
        rowStart = len(rows)
    return rows, cols