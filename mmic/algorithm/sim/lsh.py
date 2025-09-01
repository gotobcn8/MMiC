from torchtext.data import get_tokenizer
import numpy as np
import torch
from torch import tensor
from numba import guvectorize
from numba import float64, intp
# This algorithm is imitate 
def tokenization():
    tokenizer = get_tokenizer('basic_english')
    tokens = tokenizer("what the fuck,kcs,ooo,ggggtt")
    print(tokens)

def words_to_vocab(data):
    a =set(data)
    

#first init using Gaussin
class SignRandomProjections():
    def __init__(self,each_hash_num,data_volume,
                 data_dimension,random_seed = 777) -> None:
        self.total_num_hashes = each_hash_num * data_volume
        self.dimension = data_dimension
        self.hash_num = each_hash_num
        self.data_rows = data_volume
        np.random.seed(random_seed)
        #it's a random Gaussin distributed matrix
        self.GaussinDist = torch.from_numpy(
            np.random.normal(
                size=(self.total_num_hashes,self.dimension)
            )
        ).float()
        #for better reflect and avoiding repetition value
        self.pow2 = torch.from_numpy(
            np.array(
                [2**i for i in range(self.hash_num)]
            )
        ).float()
    
    def hash(self,x,device):
        if x.dtype != torch.float:
            x = x.float()
        #get signature
        signature = torch.sign(
            torch.matmul(
                x.to(device),self.GaussinDist.to(device).T
            )
        )
        # signature transfer to 0 or 1
        binsignature = torch.gt(signature,0).float()
        # finally reshape
        binsignature = binsignature.reshape(-1,self.data_rows,self.hash_num)
        return torch.matmul(
            binsignature,
            self.pow2.to(device)
        ).int().cpu().numpy()
        
        
class ReflectSketch():
    def __init__(self,hashF,dtype,data_vol,hash_num,dimension) -> None:
        self.dtype = dtype
        self.sketch_rows = data_vol
        #increase the sketch range, not easy to repeat
        self.sketch_cols = 2 ** hash_num
        self.dimension = dimension
        self.NumberData = 0
        self.sketch = np.zeros(
            (self.sketch_rows,self.sketch_cols),
            dtype=self.dtype,
        )
        self.hashF = hashF
    
    def get_sketch(self,X,device):
        self.NumberData += X.shape[0]
        if X.dim() > 2:
            X = X.reshape(len(X),-1)
        hashcodes = self.hashF.hash(X,device)
        # if device == 'cuda':
        #     hashcodes = hashcodes.cpu().numpy()
        self.increasecount(hashcodes)
    
    # def multimodal_get_sketch(self,x,device):
        
    
    # @guvectorize([( intp[:,:], float64[:], float64[:,:], float64[:,:])], '(n,l1),(l2),(m,k)->(m,k)',
    #              target = "parallel", nopython=True,cache = True)
    def increasecount(self,hashcodes):
        for i in range(self.sketch.shape[0]):
            for j in range(hashcodes.shape[0]):
                self.sketch[i, hashcodes[j,i]] += 1.0