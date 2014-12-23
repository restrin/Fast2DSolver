# Elimination Tree
class Node:
    def __init__(self, ix):
        self.ix = ix;
        # Sets are slow, will manually enforce avoiding duplicate children
        self.children = []
        # Root has parent -1
        self.parent = -1
    
    def addChild(self, c):
        self.children.append(c)
        
    def getChildren(self):
        return self.children
    
    # Parent nodes are for printing
    def addParent(self, p):
        self.parent = p;
        
    def getParent(self):
        return self.parent

class EliminationTree:
    def __init__(self, n):
        self.n = n
        self.nodes = map(Node, range(n*n))
        self.root = n*n-1;   
                
    def getRoot(self):
        return self.nodes[self.root];
        
    def getChildren(self, node):
        # Returns list of children node objects
        res = [];
        for c in self.nodes[node].getChildren():
            res.append(self.nodes[c])
        return res;
    
    def getNode(self, node):
        return self.nodes[node]
        
    def printTree(self):
        for node in self.nodes:
            print node.ix, node.getParent()

# Supernodal Elimination Tree
class SuperNode:
    def __init__(self, superNodeId):
        self.ix = superNodeId
        self.nodes = []
        self.children = []
        self.parent = -1
        
        self.A_matrix_tree = False
        # (1,1)-block of Frontal matrix
        self.A_matrix = None
        # (1,2)-block of Frontal matrix is B1*B2
        self.B1_matrix = None
        self.B2_matrix = None
        # (2,1)-block of Frontal matrix is C1*C2
        self.C1_matrix = None
        self.C2_matrix = None
        # Indices of Frontal matrix
        # Only contains those indices not in self.nodes
        self.frontalIndices = []
        # Partial solution vector for nodes in self.nodes
        self.soln = None
        # Contribution of solution vector to parents, used during solve
        # Is just C1*C2*soln
        self.parentContribution = None
    
    def setNodes(self, nodes):
        self.nodes = nodes
       
    def addChild(self, c):
        self.children.append(c)
        
    def getChildren(self):
        return self.children
    
    # Parent nodes are for printing
    def addParent(self, p):
        self.parent = p;
        
    def getParent(self):
        return self.parent

class SupernodalEliminationTree(EliminationTree):
    def __init__(self, numberOfSupernodes, numberOfNodes):
        self.numberOfSupernodes = numberOfSupernodes
        self.numberOfNodes = numberOfNodes
        self.nodes = map(SuperNode, range(numberOfSupernodes))
        self.root = numberOfSupernodes - 1;

def buildSupernodalEliminationTreeRecursive(M, levels, minNodeSize):
    etree = SupernodalEliminationTree(levels, len(M)*len(M[0]))
    buildSupernodalEliminationTreeHelper(etree, M, levels-1, 1, -1, minNodeSize)
    return etree

def buildSupernodalEliminationTreeHelper(etree, M, currentNode, direction, parent, minNodeSize):
    if (M.shape[direction] <= minNodeSize):
        addSupernodalEdges(etree, currentNode, M[0,0], M[-1,-1], parent);
        return currentNode
    else:
        n = M.shape[direction];
        newParent = currentNode
        if (direction == 0):
            addSupernodalEdges(etree, currentNode, M[n/2,0], M[n/2,-1], parent);
            currentNode = buildSupernodalEliminationTreeHelper(etree, M[n/2+1:,:], currentNode-1, (direction+1)%2, newParent, minNodeSize);
            return buildSupernodalEliminationTreeHelper(etree, M[0:n/2,:], currentNode-1, (direction+1)%2, newParent, minNodeSize);
        else:
            addSupernodalEdges(etree, currentNode, M[0,n/2], M[-1,n/2], parent);
            currentNode = buildSupernodalEliminationTreeHelper(etree, M[:,n/2+1:], currentNode-1, (direction+1)%2, newParent, minNodeSize);
            return buildSupernodalEliminationTreeHelper(etree, M[:,0:n/2], currentNode-1, (direction+1)%2, newParent, minNodeSize);

def addSupernodalEdges(etree, currentNode, first, last, parent):
    node = etree.getNode(currentNode)
    node.setNodes(xrange(first, last+1))
    node.addParent(parent)
    if (parent != -1):
        etree.getNode(parent).addChild(currentNode)



################################################################################
# Older stuff

# M is the mesh already ordered by Nested Dissection
def buildEliminationTreeRecursive(M):
    etree = EliminationTree(len(M));
    buildEliminationTreeHelper(etree, M, 1, -1);
    return etree

def buildEliminationTreeHelper(etree, M, direction, parent):
    if (M.shape[direction] <= 2):
        addEdges(etree, M[0,0], M[-1,-1], parent);
    else:
        n = M.shape[direction];
        if (direction == 0):
            addEdges(etree, M[n/2,0], M[n/2,-1], parent);
            buildEliminationTreeHelper(etree, M[0:n/2,:], (direction+1)%2, M[n/2,0]);
            buildEliminationTreeHelper(etree, M[n/2+1:,:], (direction+1)%2, M[n/2,0]);
        else:
            addEdges(etree, M[0,n/2], M[-1,n/2], parent);
            buildEliminationTreeHelper(etree, M[:,0:n/2], (direction+1)%2, M[0,n/2]);
            buildEliminationTreeHelper(etree, M[:,n/2+1:], (direction+1)%2, M[0,n/2]);
            
def addEdges(etree, first, last, parent):
    for j in range(last,first,-1):
        etree.getNode(j-1).addParent(j)
        etree.getNode(j).addChild(j-1)
    etree.getNode(last).addParent(parent)
    if (parent != -1):
        # Root has no child to add
        etree.getNode(parent).addChild(last)
 
    
### Direct way this way doesn't work, needs to involve some complications.
### Use recursive instead.   
def getNeighbours(M, i, j):
    neighbours = []
    indices = [[i],[j]]
    if (i - 1 >= 0):
        indices[0].append(i-1);
    if (i + 1 < len(M)):
        indices[0].append(i+1);
    if (j - 1 >= 0):
        indices[1].append(j-1);
    if (j + 1 < len(M)):
        indices[1].append(j+1);
    for ix in indices[0]:
        for jx in indices[1]:
            neighbours.append(M[ix,jx]);
    return neighbours
               
def buildEliminationTreeDirect(M):
    etree = EliminationTree(len(M))
    for i in range(len(M)):
        for j in range(len(M[0])):
            child = M[i,j];
            neighbours = getNeighbours(M,i,j)
            # The minimum neighbour larger than child is the parent
            largerNeighbours = filter(lambda x: x > child, neighbours)
            if (len(largerNeighbours) == 0):
                # Biggest node number, must be root
                continue;
            parent = min(largerNeighbours);
            etree.getNode(parent).addChild(child)
            etree.getNode(child).addParent(parent)
    return etree