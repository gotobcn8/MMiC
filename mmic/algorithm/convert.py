import numpy as np
import torch

def to_numpy(tensor:torch.tensor, n_dims=2):
    """Convert a torch tensor to numpy array.

    Args:
        tensor (Tensor): a tensor object to convert.
        n_dims (int): size of numpy array shape
    """
    try:
        nparray = tensor.detach().cpu().clone().numpy()
    except AttributeError:
        raise TypeError('tensor type should be torch.Tensor, not {}'.format(type(tensor)))

    while len(nparray.shape) < n_dims:
        nparray = np.expand_dims(nparray, axis=0)

    return nparray
