import numpy as np

def nestedDissectionOrdering2(P, level, direction):
    if (P.shape[direction] <= 2):
        P[:,:] = level;
        return level
    else:
        n = P.shape[direction];
        if (direction == 0):
            P[n/2,:] = level;
            level = nestedDissectionOrdering2(P[0:n/2,:], level - 1, (direction+1)%2);
            return nestedDissectionOrdering2(P[n/2+1:,:], level - 1, (direction+1)%2);
        else:
            P[:,n/2] = level;
            level = nestedDissectionOrdering2(P[:,0:n/2], level-1, (direction+1)%2);
            return nestedDissectionOrdering2(P[:,n/2+1:], level-1, (direction+1)%2);  