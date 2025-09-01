import torch

def reshape_with(x,target_dimension):
    '''
    This is reshape(x) to (-1,dimension)\n
    If is not as the shape(-1,dimension), we will try to fill with zeros
    '''
    dim_nums = x.numel()
    redundant = dim_nums % target_dimension
    # If corresponding to target dimension
    if redundant == 0:
        return x.reshape(-1,target_dimension)
    x = x.reshape(1,-1)
    padding_size = target_dimension - redundant
    x = torch.cat([x, torch.zeros((1,padding_size), dtype=x.dtype)],dim = 1)
    return x.reshape(-1,target_dimension)