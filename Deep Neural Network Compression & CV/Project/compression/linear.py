import torch
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
class modified_linear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
    def prune(self,threshold):
        #this function should prune the weights which are below threshold
        pass
    def quantize(self,k):
        '''cluster the weights in k different cluster, and then use 2 things, 
        one is a cluster map which will be o(nxn) uint8 type and one array of clusters which wil
        be a O(k) sized fload 34 

        note that SK-learn is a cpu library!
        '''
        pass
    def forward(self, input):
        if(self.mode == 'normal'):
            print("hello")
        elif(self.mode == 'prune'):
            print("hello")
        elif(self.mode == 'quantize'):
            print("hello")
        return

    
