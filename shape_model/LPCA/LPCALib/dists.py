import numpy as np
from abc import ABC, abstractmethod
 
class DistBase(ABC):
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def dist(self, x, y):
        pass

class SimpleMatrixDist(DistBase):
    def __init__(self, dist_matrix):
        super().__init__()
        self.dist_matrix=dist_matrix

    def dist(self, x, y):
        return self.dist_matrix[np.ix_(x,y)]

class ShapeNDEuclideanDist(DistBase):
    def __init__(self, points):
        super().__init__()
        self.points=np.array(points)

    def dist(self, x, y):
        return np.sqrt(np.sum((np.repeat(self.points[x,:][:,np.newaxis,:],len(y),axis=1)-np.repeat(self.points[y,:][np.newaxis,:,:],len(x),axis=0))**2,axis=2))

class Field3DEuclideanDist(DistBase):
    def __init__(self, pnt_idx):
        super().__init__()
        self.pnt_idx=pnt_idx

    def dist(self, x, y):
        print('shape x: '+str(x.shape)+' shape y: '+str(y.shape))
        return np.sqrt(np.sum((np.repeat(self.pnt_idx[x,:][:,np.newaxis,:],len(y),axis=1)-np.repeat(self.pnt_idx[y,:][np.newaxis,:,:],len(x),axis=0))**2,axis=2))
#        return np.sum((np.repeat(self.pnt_idx[x,:][:,np.newaxis,:],len(y),axis=1)-np.repeat(self.pnt_idx[y,:][np.newaxis,:,:],len(x),axis=0))**2,axis=2)
        #dist_matrix=np.zeros((len(x),len(y)))
        #slow! should be optimized!
        #for i in range(0,len(x)):
        #    for j in range(0,len(y)):    
        #        dist_matrix[i,j]=np.sqrt(np.sum((self.pnt_idx[x[i],:]-self.pnt_idx[y[j],:])**2))
        #return dist_matrix
