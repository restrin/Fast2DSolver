import numpy as np
import scipy.linalg as li
import scipy.sparse as sp
import matplotlib.pylab as pl
import scipy.sparse.linalg as spli

class HODLRNode:
    def __init__(self):
        self.leftChild = None
        self.rightChild = None
        
        self.K = None
        self.U1 = None
        self.V1 = None
        self.U2 = None
        self.V2 = None
        self.d1 = None
        self.d2 = None
        self.V1d2 = None
        self.V2d1 = None
        self.LU = None

def HODLRDirectSolverFactored( HODLRNode, F):
    
    # Base Case
    if (HODLRNode.leftChild == None):
        return li.solve(HODLRNode.K, F)
    else:
        
        p = HODLRNode.U1.shape[0]
        
        U1 = HODLRNode.U1
        U2 = HODLRNode.U2
        V1 = HODLRNode.V1
        V2 = HODLRNode.V2
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        
        c1 = HODLRDirectSolverFactored(HODLRNode.leftChild, F[:p,:])
        c2 = HODLRDirectSolverFactored(HODLRNode.rightChild, F[p:,:])
        
        d1 = HODLRNode.d1
        d2 = HODLRNode.d2
        V2d1 = HODLRNode.V2d1
        V1d2 = HODLRNode.V1d2
        S1 = np.concatenate((np.eye(r2), V2d1),1)
        S2 = np.concatenate((V1d2, np.eye(r1)),1)
        S = np.concatenate((S1,S2),0)
        
        y = li.solve(S, np.concatenate((np.dot(V2,c1), np.dot(V1,c2)),0))
        
        y1 = y[:r2,:]
        y2 = y[r2:,:]
        
        x1 = c1 - np.dot(d1,y2)
        x2 = c2 - np.dot(d2,y1)
        
        return np.concatenate((x1, x2), 0)

def HODLRDirectSolverFactoredSMW( HODLRNode, F):
    
    # Base Case
    if (HODLRNode.leftChild == None):
        return li.solve(HODLRNode.K, F)
    else:
        
        p = HODLRNode.U1.shape[0]
        
        V1 = HODLRNode.V1
        V2 = HODLRNode.V2
        r1 = V1.shape[0]
        
        c1 = HODLRDirectSolverFactoredSMW(HODLRNode.leftChild, F[:p,:])
        c2 = HODLRDirectSolverFactoredSMW(HODLRNode.rightChild, F[p:,:])
        
        d1 = HODLRNode.d1
        d2 = HODLRNode.d2
#        S1 = np.concatenate((np.eye(r2), np.dot(V2, d1)),1)
#        S2 = np.concatenate((np.dot(V1,d2), np.eye(r1)),1)
#        S = np.concatenate((S1,S2),0)
        
#        y = li.solve(S, np.concatenate((np.dot(V2,c1), np.dot(V1,c2)),0))
        
#        y1 = y[:r2,:]
#        y2 = y[r2:,:]
        
        V2d1 = HODLRNode.V2d1
        V1d2 = HODLRNode.V1d2
        
        # Forward substitution
        y1 = np.dot(V2,c1)
        y2 = np.dot(V1,c2) - np.dot(V1d2, y1)
        # Backward substitution
#        y2 = li.solve(np.eye(r1) - np.dot(V1d2, V2d1), y2)
        y2 = li.lu_solve(HODLRNode.LU, y2)
        y1 = y1 - np.dot(V2d1,y2)
        
        x1 = c1 - np.dot(d1,y2)
        x2 = c2 - np.dot(d2,y1)
        
        return np.concatenate((x1, x2), 0)

def HODLRDirectSolverBuildTreeSMW( K, F, minSize, tol ):
    
    hodlrNode = HODLRNode()
    
    n = K.shape[0]
    
    # Base Case
    if (n <= minSize):
        hodlrNode.K = K
        return li.solve(K, F), hodlrNode
    else:
        p = (n/2)
        U1,V1 = lowRankApprox(K[:p,p:], -1, tol)
        U2,V2 = lowRankApprox(K[p:,:p], -1, tol)
        
        hodlrNode.U1 = U1
        hodlrNode.V1 = V1
        hodlrNode.U2 = U2
        hodlrNode.V2 = V2
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        
        d1c1, leftChild = HODLRDirectSolverBuildTree(K[:p,:p], np.concatenate((U1, F[:p,:]),1), minSize, tol)
        d2c2, rightChild = HODLRDirectSolverBuildTree(K[p:,p:], np.concatenate((U2, F[p:,:]),1), minSize, tol)
        
        hodlrNode.leftChild = leftChild
        hodlrNode.rightChild = rightChild
        
        d1 = d1c1[:,:r1]
        c1 = d1c1[:,r1:]
        d2 = d2c2[:,:r2]
        c2 = d2c2[:,r2:]
        
        hodlrNode.d1 = d1
        hodlrNode.d2 = d2
        
        hodlrNode.V1d2 = np.dot(V1,d2)
        hodlrNode.V2d1 = np.dot(V2,d1)

        # Forward substitution
        y1 = np.dot(V2,c1)
        y2 = np.dot(V1,c2) - np.dot(V1, np.dot(d2, y1))
        # Backward substitution
        hodlrNode.LU = li.lu_factor(np.eye(r1) - np.dot(np.dot(V1, d2), np.dot(V2,d1)));
        y2 = li.lu_solve(hodlrNode.LU, y2)
        y1 = y1 - np.dot(V2,np.dot(d1,y2))
        
        x1 = c1 - np.dot(d1,y2)
        x2 = c2 - np.dot(d2,y1)
        
        return np.concatenate((x1, x2), 0), hodlrNode

def HODLRDirectSolverBuildTree( K, F, minSize, tol ):
    
    hodlrNode = HODLRNode()
    
    n = K.shape[0]
    
    # Base Case
    if (n <= minSize):
        hodlrNode.K = K
        return li.solve(K, F), hodlrNode
    else:
        p = (n/2)
        U1,V1 = lowRankApprox(K[:p,p:], -1, tol)
        U2,V2 = lowRankApprox(K[p:,:p], -1, tol)
        
        hodlrNode.U1 = U1
        hodlrNode.V1 = V1
        hodlrNode.U2 = U2
        hodlrNode.V2 = V2
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        
        d1c1, leftChild = HODLRDirectSolverBuildTree(K[:p,:p], np.concatenate((U1, F[:p,:]),1), minSize, tol)
        d2c2, rightChild = HODLRDirectSolverBuildTree(K[p:,p:], np.concatenate((U2, F[p:,:]),1), minSize, tol)
        
        hodlrNode.leftChild = leftChild
        hodlrNode.rightChild = rightChild
        
        d1 = d1c1[:,:r1]
        c1 = d1c1[:,r1:]
        d2 = d2c2[:,:r2]
        c2 = d2c2[:,r2:]
        
        hodlrNode.d1 = d1
        hodlrNode.d2 = d2
        
        hodlrNode.V1d2 = np.dot(V1,d2)
        hodlrNode.V2d1 = np.dot(V2,d1)
        S1 = np.concatenate((np.eye(r2), hodlrNode.V2d1),1)
        S2 = np.concatenate((hodlrNode.V1d2, np.eye(r1)),1)
        S = np.concatenate((S1,S2),0)
        
        y = li.solve(S, np.concatenate((np.dot(V2,c1), np.dot(V1,c2)),0))
        
        y1 = y[:r2,:]
        y2 = y[r2:,:]
        
        x1 = c1 - np.dot(d1,y2)
        x2 = c2 - np.dot(d2,y1)
        
        return np.concatenate((x1, x2), 0), hodlrNode

def lowRankApprox(K, r, tol):
    if (K.shape[0] == 0 or K.shape[1] == 0):
        return None, None
    # SVD returns V' instead of V for some silly reason
    U, S, V = li.svd(K)
    if (r < 0):
        S = filter(lambda x: x > tol, S)
        r = len(S)
    
    return S[:r]*U[:,:r], V[:r,:]
    
def HODLRDirectSolver( K, F, minSize, tol ):
    n = K.shape[0]
    
    # Base Case
    if (n <= minSize):
        return li.solve(K, F)
    else:
        p = (n/2)
        U1,V1 = lowRankApprox(K[:p,p:], -1, tol)
        U2,V2 = lowRankApprox(K[p:,:p], -1, tol)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        
        d1c1 = HODLRDirectSolver(K[:p,:p], np.concatenate((U1, F[:p,:]),1), minSize, tol)
        d2c2 = HODLRDirectSolver(K[p:,p:], np.concatenate((U2, F[p:,:]),1), minSize, tol)
        
        d1 = d1c1[:,:r1]
        c1 = d1c1[:,r1:]
        d2 = d2c2[:,:r2]
        c2 = d2c2[:,r2:]
        
        S1 = np.concatenate((np.eye(r2), np.dot(V2, d1)),1)
        S2 = np.concatenate((np.dot(V1,d2), np.eye(r1)),1)
        S = np.concatenate((S1,S2),0)
        
        y = li.solve(S, np.concatenate((np.dot(V2,c1), np.dot(V1,c2)),0))
        
        y1 = y[:r2,:]
        y2 = y[r2:,:]
        
        x1 = c1 - np.dot(d1,y2)
        x2 = c2 - np.dot(d2,y1)
        
        return np.concatenate((x1, x2), 0)

def HODLRDirectSolverSMW( K, F, minSize, tol ):
    n = K.shape[0]
    
    # Base Case
    if (n <= minSize):
        return li.solve(K, F)
    else:
        p = (n/2)
        U1,V1 = lowRankApprox(K[:p,p:], -1, tol)
        U2,V2 = lowRankApprox(K[p:,:p], -1, tol)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        
        d1c1 = HODLRDirectSolverSMW(K[:p,:p], np.concatenate((U1, F[:p,:]),1), minSize, tol)
        d2c2 = HODLRDirectSolverSMW(K[p:,p:], np.concatenate((U2, F[p:,:]),1), minSize, tol)
        
        d1 = d1c1[:,:r1]
        c1 = d1c1[:,r1:]
        d2 = d2c2[:,:r2]
        c2 = d2c2[:,r2:]
                                                
#        S1 = np.concatenate((np.eye(r2), np.dot(V2, d1)),1)
#        S2 = np.concatenate((np.dot(V1,d2), np.eye(r1)),1)
#        S = np.concatenate((S1,S2),0)
        
#        y = li.solve(S, np.concatenate((np.dot(V2,c1), np.dot(V1,c2)),0))
        
#        y1 = y[:r2,:]
#        y2 = y[r2:,:]

        # Forward substitution
        y1 = np.dot(V2,c1)
        y2 = np.dot(V1,c2) - np.dot(V1, np.dot(d2, y1))
        # Backward substitution
        y2 = li.solve(np.eye(r1) - np.dot(np.dot(V1, d2), np.dot(V2,d1)), y2)
        y1 = y1 - np.dot(V2,np.dot(d1,y2))
                        
        x1 = c1 - np.dot(d1,y2)
        x2 = c2 - np.dot(d2,y1)
        
        return np.concatenate((x1, x2), 0)
        
def HODLRDirectSolverSMWReturnRanks( K, F, minSize, tol ):
    n = K.shape[0]
    
    ranks = []
    
    # Base Case
    if (n <= minSize):
        return li.solve(K, F), []
    else:
        p = (n/2)
        U1,V1 = lowRankApprox(K[:p,p:], -1, tol)
        U2,V2 = lowRankApprox(K[p:,:p], -1, tol)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        
        ranks.append(r1)
        ranks.append(r2)
        
        d1c1, ranks1 = HODLRDirectSolverSMWReturnRanks(K[:p,:p], np.concatenate((U1, F[:p,:]),1), minSize, tol)
        d2c2, ranks2 = HODLRDirectSolverSMWReturnRanks(K[p:,p:], np.concatenate((U2, F[p:,:]),1), minSize, tol)
        
        ranks = ranks + ranks1
        ranks = ranks + ranks2
        
        d1 = d1c1[:,:r1]
        c1 = d1c1[:,r1:]
        d2 = d2c2[:,:r2]
        c2 = d2c2[:,r2:]
                                                
#        S1 = np.concatenate((np.eye(r2), np.dot(V2, d1)),1)
#        S2 = np.concatenate((np.dot(V1,d2), np.eye(r1)),1)
#        S = np.concatenate((S1,S2),0)
        
#        y = li.solve(S, np.concatenate((np.dot(V2,c1), np.dot(V1,c2)),0))
        
#        y1 = y[:r2,:]
#        y2 = y[r2:,:]

        # Forward substitution
        y1 = np.dot(V2,c1)
        y2 = np.dot(V1,c2) - np.dot(V1, np.dot(d2, y1))
        # Backward substitution
        y2 = li.solve(np.eye(r1) - np.dot(np.dot(V1, d2), np.dot(V2,d1)), y2)
        y1 = y1 - np.dot(V2,np.dot(d1,y2))
                        
        x1 = c1 - np.dot(d1,y2)
        x2 = c2 - np.dot(d2,y1)
        
        return np.concatenate((x1, x2), 0), ranks