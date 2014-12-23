import numpy as np
import scipy.linalg as li
import scipy.sparse as sp
import matplotlib.pylab as pl
import scipy.sparse.linalg as spli
import bisect
from symbChol import *
from eliminationTree import *
from hodlrDirectSolver import *

### Methods related to constructing solution vector

def solveTree(etree, b, approxTol):
    forwardSolve(etree, b)
    x = constructSolutionVector(etree)
    backwardSolve(etree, x, approxTol)
    return constructSolutionVector(etree)

def solveTreeEliminationTree(etree, b, approxTol):
    forwardSolve(etree, b)
    backwardSolveEliminationTree(etree, approxTol)
    return constructSolutionVector(etree)

def solveTreeEliminationTree2(etree, b, approxTol):
    forwardSolve2(etree, b, approxTol)
    backwardSolveEliminationTree2(etree)
    return constructSolutionVector(etree)

def constructSolutionVector(etree):
    n = etree.numberOfNodes
    x = np.zeros((n,1))
    for j in xrange(etree.numberOfSupernodes):
        supernode = etree.getNode(j)
        x[supernode.nodes] = supernode.soln
    return x

def forwardSolve(etree, b):
    for j in xrange(etree.numberOfSupernodes):
        supernode = etree.getNode(j)
        t = supernode.nodes[-1] - supernode.nodes[0] + 1
        soln = -b[supernode.nodes[0]:supernode.nodes[0]+t]
        soln = np.concatenate((soln, np.zeros((len(supernode.frontalIndices),1))),0)
        pInd = np.concatenate((supernode.nodes, supernode.frontalIndices),0)
        
        # Indices for children nodes, actual nodes stored in tree
        for c in supernode.getChildren():
            child = etree.getNode(c)
            soln, pInd = vectorExtendAdd(soln, pInd, \
                child.parentContribution, child.frontalIndices)          
        
        supernode.soln = -soln[:t]
        
        if (j < etree.numberOfSupernodes-1):
            pContrib = soln[t:]
            supernode.parentContribution, supernode.frontalIndices = \
            vectorExtendAdd( \
                pContrib, \
                pInd[t:], \
                np.dot(supernode.C1_matrix, np.dot(supernode.C2_matrix, supernode.soln)), \
                supernode.frontalIndices)

def forwardSolve2(etree, b, approxTol):
    for j in xrange(etree.numberOfSupernodes):
        supernode = etree.getNode(j)
        t = supernode.nodes[-1] - supernode.nodes[0] + 1
        soln = -b[supernode.nodes[0]:supernode.nodes[0]+t]
        soln = np.concatenate((soln, np.zeros((len(supernode.frontalIndices),1))),0)
        pInd = np.concatenate((supernode.nodes, supernode.frontalIndices),0)
        
        # Indices for children nodes, actual nodes stored in tree
        for c in supernode.getChildren():
            child = etree.getNode(c)
            soln, pInd = vectorExtendAdd(soln, pInd, \
                child.parentContribution, child.frontalIndices)          
        
        supernode.soln = -soln[:t]
        
        if (supernode.A_matrix_tree):
            supernode.soln = HODLRDirectSolverFactoredSMW(supernode.A_matrix, supernode.soln)
        else:
            supernode.soln, supernode.A_matrix = HODLRDirectSolverBuildTreeSMW(supernode.A_matrix, supernode.soln, int(supernode.A_matrix.shape[0]/1.5), approxTol)
            supernode.A_matrix_tree = True
        
        if (j < etree.numberOfSupernodes-1):
            pContrib = soln[t:]
            supernode.parentContribution, supernode.frontalIndices = \
            vectorExtendAdd( \
                pContrib, \
                pInd[t:], \
                np.dot(supernode.C1_matrix, np.dot(supernode.C2_matrix, supernode.soln)), \
                supernode.frontalIndices)


# Doesn't use explicitly constructed right-hand side
# Unit upper-triangular
def backwardSolveEliminationTree2(etree):    
    for j in xrange(etree.numberOfSupernodes-1,-1,-1):
        supernode = etree.getNode(j)
        t = supernode.nodes[-1] - supernode.nodes[0] + 1
        soln = supernode.soln
        indF = np.array(supernode.frontalIndices)
        nodeId = supernode.getParent()
        ctr = 0
        total_updates = len(indF)
        
        while (ctr < total_updates):
            node = etree.getNode(nodeId)
            nodes = node.nodes
                        
            if (nodes[-1] < indF[ctr]):
                nodeId = node.getParent()
                continue
            
            nt = len(filter(lambda x: x <= nodes[-1],indF[ctr:]))
            parent_first = bisect.bisect_left(nodes,indF[ctr])
            soln = soln - np.dot(supernode.B1_matrix, \
                np.dot(supernode.B2_matrix[:,ctr:ctr+nt], \
                node.soln[parent_first:parent_first+nt]))
            nodeId = node.getParent()
            ctr = ctr + nt

        supernode.soln = soln

# Doesn't use explicitly constructed right-hand side
def backwardSolveEliminationTree(etree, approxTol):    
    for j in xrange(etree.numberOfSupernodes-1,-1,-1):
        supernode = etree.getNode(j)
        t = supernode.nodes[-1] - supernode.nodes[0] + 1
        soln = supernode.soln
        indF = np.array(supernode.frontalIndices)
        nodeId = supernode.getParent()
        ctr = 0
        total_updates = len(indF)
        
        while (ctr < total_updates):
            node = etree.getNode(nodeId)
            nodes = node.nodes
                        
            if (nodes[-1] < indF[ctr]):
                nodeId = node.getParent()
                continue
            
            nt = len(filter(lambda x: x <= nodes[-1],indF[ctr:]))
            parent_first = bisect.bisect_left(nodes,indF[ctr])
            soln = soln - np.dot(supernode.B1_matrix, \
                np.dot(supernode.B2_matrix[:,ctr:ctr+nt], \
                node.soln[parent_first:parent_first+nt]))
            nodeId = node.getParent()
            ctr = ctr + nt

        if (supernode.A_matrix_tree):
            supernode.soln = HODLRDirectSolverFactoredSMW(supernode.A_matrix, soln)
        else:
            supernode.soln, supernode.A_matrix = HODLRDirectSolverBuildTreeSMW(supernode.A_matrix, soln, int(supernode.A_matrix.shape[0]/1.5), approxTol)
            supernode.A_matrix_tree = True
                        
def backwardSolve(etree, b, approxTol):
    for j in xrange(etree.numberOfSupernodes-1,-1,-1):
        supernode = etree.getNode(j)
        t = supernode.nodes[-1] - supernode.nodes[0] + 1
        soln = b[supernode.nodes[0]:supernode.nodes[0]+t]
        indF = np.array(supernode.frontalIndices)
        nodeId = supernode.getParent()
        ctr = 0
        total_updates = len(indF)
        
        while (ctr < total_updates):
            node = etree.getNode(nodeId)
            nodes = node.nodes
                        
            if (nodes[-1] < indF[ctr]):
                nodeId = node.getParent()
                continue
            
            nt = len(filter(lambda x: x <= nodes[-1],indF[ctr:]))
            parent_first = bisect.bisect_left(nodes,indF[ctr])
            soln = soln - np.dot(supernode.B1_matrix, \
                np.dot(supernode.B2_matrix[:,ctr:ctr+nt], \
                node.soln[parent_first:parent_first+nt]))
            nodeId = node.getParent()
            ctr = ctr + nt

        if (supernode.A_matrix_tree):
            supernode.soln = HODLRDirectSolverFactoredSMW(supernode.A_matrix, soln)
        else:
            supernode.soln, supernode.A_matrix = HODLRDirectSolverBuildTreeSMW(supernode.A_matrix, soln, 10, approxTol)
            supernode.A_matrix_tree = True

def vectorExtendAdd(v1, ind1, v2, ind2):
    indUnion = sorted(list(set(ind1).union(set(ind2))))
    return vectorExtend(v1, ind1, indUnion) + vectorExtend(v2, ind2, indUnion), indUnion
    
def vectorExtend(v1, ind1, ind_union):
    res = np.zeros((len(ind_union),1))
    
    ix = [0 for i in xrange(len(ind1))]
    for i in xrange(len(ind1)):
        ix[i] = bisect.bisect_left(ind_union,ind1[i])
    
    for i in xrange(len(ind1)):
#        res[bisect.bisect_left(ind_union,i)] = v1[bisect.bisect_left(ind1,i)]
        res[ix[i]] = v1[i]
    return res      



### Methods related to doing supernodal multifrontal
def supernodalMultifrontal(A, M, levels, minNodeSize, approxTol):
    # Building supernodal version
    etree = buildSupernodalEliminationTreeRecursive(M, levels, minNodeSize)
    # G is CSC sparse matrix
    G = supernodalSymbChol(A, etree)
    
    A = A.tolil()
#    L = np.zeros(A.shape)
#    U = np.zeros(A.shape)
    
    # For now store the update matrices in a list (stack)
    # Parallel list stores the indices that the list corresponds to
    updateMatrices = []
    updateIndices = []
    
    for j in xrange(levels):
        supernode = etree.getNode(j)
        children = supernode.getChildren()

        # Get Fj
        Fj, indF, t = getSupernodalFrontalMatrix(A, G, j, etree)                                                                   
                                                                                                                                   
        # Do Extend-Add for all children's update matrices        
        for c in children:
            Fj, indF = extendAdd(Fj, indF, updateMatrices.pop(), updateIndices.pop())
        
#        supernode.A_matrix = Fj[:t,:t]
        
#        U[np.ix_(supernode.nodes, supernode.nodes)] = np.identity(len(supernode.nodes))
#        L[np.ix_(supernode.nodes, supernode.nodes)] = Fj[:t,:t]
        
        # If this is root node, we don't need to do anything further
        if (j == levels-1):
            supernode.A_matrix = Fj[:t,:t]
            supernode.A_matrix_tree = False
            continue
          
        B1, B2 = lowRankApprox(Fj[:t,t:], -1, approxTol)            
        C1, C2 = lowRankApprox(Fj[t:,:t], -1, approxTol)
        supernode.B1_matrix = B1
        supernode.B2_matrix = B2
        supernode.C1_matrix = C1
        supernode.C2_matrix = C2

        # Factor F
        factorFastDirect2(etree.nodes[j], Fj)
        
#        if (j != levels-1):
#            L[np.ix_(indF[t:], supernode.nodes)] = Fj[t:,:t]        
#            U[np.ix_(supernode.nodes, list(indF[t:]))] = li.solve(Fj[:t,:t], Fj[:t,t:])
#            L[np.ix_(indF[t:], supernode.nodes)] = np.transpose(li.solve(np.transpose(Fj[:t,:t]), np.transpose(Fj[t:,:t])))        
#            U[np.ix_(supernode.nodes, list(indF[t:]))] = Fj[:t,t:]
        
        supernode.frontalIndices = indF[t:]
        
        # Push onto stack
        updateMatrices.append(Fj[t:,t:])
        updateIndices.append(indF[t:])    
                          
    return etree#, L, U

# Makes unit upper triangular matrix, instead of lower
def factorFastDirect2(supernode, Fj):
    t = supernode.nodes[-1] - supernode.nodes[0] + 1

    supernode.B1_matrix = HODLRDirectSolverSMW( \
        Fj[:t,:t], supernode.B1_matrix, 50, 1e-8 )

    supernode.A_matrix = Fj[:t,:t]
#    supernode.A_matrix_tree = True

    Fj[t:,t:] = Fj[t:,t:] - np.dot(np.dot(Fj[t:,:t], supernode.B1_matrix), \
        supernode.B2_matrix)

def factorFastDirect(supernode, Fj):
    t = supernode.nodes[-1] - supernode.nodes[0] + 1

    # Need to sort out issue with transpose
    C2 = HODLRDirectSolverSMW( \
        np.transpose(Fj[:t,:t]), np.transpose(supernode.C2_matrix), 50, 1e-8)

    supernode.A_matrix = Fj[:t,:t]
#    supernode.A_matrix_tree = True
    supernode.C2_matrix = np.transpose( C2 )

#   Commented out old solving code
#    supernode.C2_matrix = np.transpose( \
#        li.solve(np.transpose(supernode.A_matrix), \
#        np.transpose(supernode.C2_matrix)))

    # Uj = D - (C/A)*B
    Fj[t:,t:] = Fj[t:,t:] - np.dot(supernode.C1_matrix, \
        np.dot(supernode.C2_matrix, Fj[:t,t:]))

def getSupernodalFrontalMatrix(A, L, j, etree):
    superNode = etree.getNode(j)
    first = superNode.nodes[0]
    last = superNode.nodes[-1]
    indices = L[0]
    indptr = L[1]
    rows = indices[indptr[last]+1:indptr[last+1]]#L.indices[L.indptr[last]+1:L.indptr[last+1]]
    t = last-first+1

    F = np.zeros((len(rows)+t,len(rows)+t))
    F[:t,:t] = A[first:last+1,first:last+1].todense()
    F[t:,:t] = A[rows,first:last+1].todense()
    F[:t,t:] = A[first:last+1,rows].todense()
    
#    F[:t,:t] = A.tocsc()[:,first:last+1].tocsr()[first:last+1,:].todense()
#    F[t:,:t] = A.tocsc()[:,first:last+1].tocsr()[rows,:].todense()
#    F[:t,t:] = A.tocsc()[:,rows].tocsr()[first:last+1,:].todense()

#    F[:,:t] = A[:,first:last+1].tocsr()[np.concatenate((range(first,last+1),rows)),:].todense()  
#    F[:t,t:] = A[:,rows].tocsr()[first:last+1,:].todense()
            
    return F, np.concatenate((range(first,last+1),rows),1), t

def extendAdd(U1, i1, U2, i2):
    # Assume U1, U2 are dense
    # Going to do things the slow way first. Form 2 matrices and add them
    indUnion = sorted(list(set(i1).union(set(i2))))
    return extend(U1, i1, indUnion) + extend(U2, i2, indUnion), indUnion
    
def extend(U, indices, unionIndices):
    dim = len(unionIndices)
    res = np.zeros((dim, dim))
    
    uiMap = [0 for i in indices]
    
    for i in xrange(len(indices)):
        uiMap[i] = bisect.bisect_left(unionIndices,indices[i])

    res[np.ix_(uiMap, uiMap)] = U    
            
#    for i in xrange(len(indices)):
#        for j in xrange(len(indices)):
#            res[uiMap[i], uiMap[j]] = U[i, j]
#            res[np.nonzero(unionIndices == i)[0], np.nonzero(unionIndices == j)] = \
#                U[np.nonzero(indices == i)[0], np.nonzero(indices == j)[0]];
    return res







################################################################################
### Older versions
def supernodalMultifrontalLU(A, M, levels):
    # G is CSC sparse matrix
    # Assume row indices are sorted for each column in L
    etree = buildSupernodalEliminationTreeRecursive(M, levels) #Building supernodal version
    G = supernodalSymbChol(A, etree)
    print 'Done symbolic factorization'
    
    # For now store the update matrices in a list
    # Parallel list stores the indices that the list corresponds to
#    updateMatrices = [0 for i in xrange(n)]
#    updateIndices = [[] for i in xrange(n)]
    updateMatrices = []
    updateIndices = []
    
    # L is sparse CSC matrix, shape is same as G
    L_data = np.zeros(len(G.data))
    # U is sparse CSR matrix, shape is same as G'
    U_data = np.zeros(len(G.data))
    
    ctr = 0
    
    for j in xrange(levels):
        children = etree.getNode(j).getChildren()

        # Get Fj
        Fj, indF, t = getSupernodalFrontalMatrix(A, G, j, etree)                                                                   
                                                                                                                                    
        # Do Extend-Add for all children's update matrices        
        for c in children:
            Fj, indF = extendAdd(Fj, indF, updateMatrices.pop(), updateIndices.pop())
            
        
        # Factor F
        factorSupernodalFj(Fj, t)
        
        # Save Uj
        #updateMatrices[j] = Fj[1:,1:]
        #updateIndices[j] = indF[1:]
        updateMatrices.append(Fj[t:,t:]) #NEED CHANGE
#        print 'org', j, indF[t:], indF
        updateIndices.append(indF[t:]) #NEED CHANGE
        
        # Set LU factors
        for k in xrange(t):
            L_data[ctr] = 1
            L_data[ctr+1:ctr + len(indF)-k] = Fj[k+1:,k] #NEED CHANGE
            U_data[ctr:ctr + len(indF)-k] = Fj[k,k:] #NEED CHANGE
            ctr = ctr + len(indF)-k
    
    # This probably means that if one of these matrices has row/column indices
    # mutated, it will affect the other...
    return sp.csc_matrix((L_data, G.indices, G.indptr), shape=A.shape), \
        sp.csr_matrix((U_data, G.indices, G.indptr), shape=A.shape)

# Just doing LU for now
def factorSupernodalFj(Fj, t):
    for j in xrange(t):
        Fj[j+1:,j] = Fj[j+1:,j]/Fj[j,j]  
        Fj[j+1:,j+1:] = Fj[j+1:,j+1:] - np.outer(Fj[j+1:,j],Fj[j,j+1:])

def multifrontalLU(A):
    # G is CSC sparse matrix
    # Assume row indices are sorted for each column in L
    G, etree = symbChol(A)
    print 'Done symbolic factorization'
    n = A.shape[1]
    
    # For now store the update matrices in a list
    # Parallel list stores the indices that the list corresponds to
#    updateMatrices = [0 for i in xrange(n)]
#    updateIndices = [[] for i in xrange(n)]
    updateMatrices = []
    updateIndices = []
    
    # L is sparse CSC matrix, shape is same as G
    L_data = np.zeros(len(G.data))
    # U is sparse CSR matrix, shape is same as G'
    U_data = np.zeros(len(G.data))
    
    ctr = 0
    
    for j in xrange(n):
        children = etree.getNode(j).getChildren()

        # Get Fj
        Fj, indF = getFrontalMatrix(A, G, j)                                                                    
                                                                                                                                    
        # Do Extend-Add for all children's update matrices        
        for c in children:
            Fj, indF = extendAdd(Fj, indF, updateMatrices.pop(), updateIndices.pop())
        
        # Factor F
        factorFj(Fj)
        
        # Save Uj
        #updateMatrices[j] = Fj[1:,1:]
        #updateIndices[j] = indF[1:]
        updateMatrices.append(Fj[1:,1:])
        updateIndices.append(indF[1:])
        
        # Set LU factors
        L_data[ctr] = 1
        L_data[ctr+1:ctr + len(indF)] = Fj[1:,0]
        U_data[ctr:ctr + len(indF)] = Fj[0,:]
        ctr = ctr + len(indF);
    
    # This probably means that if one of these matrices has row/column indices
    # mutated, it will affect the other...
    return sp.csc_matrix((L_data, G.indices, G.indptr), shape=(n,n)), \
        sp.csr_matrix((U_data, G.indices, G.indptr), shape=(n,n))
        

def factorFj(Fj):
    Fj[1:,0] = Fj[1:,0]/Fj[0,0]  
    Fj[1:,1:] = Fj[1:,1:] - np.outer(Fj[1:,0],Fj[0,1:])    

def getFrontalMatrix(A, L, j):
    rows = L.indices[L.indptr[j]:L.indptr[j+1]]
    F = np.zeros((len(rows),len(rows)))
    # Set the first column of F, doing stupid way for now
#    F[:,0] = A.data[A.indptr[j] + rows]
    F[:,0] = np.transpose(A[rows, j].todense())
    # Set the first row of F, using built-in slow functionality
    F[0,1:] = A[j,rows[1:]].todense()
    return F, rows
                                                                                            