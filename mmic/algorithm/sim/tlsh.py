import torch
import numpy as np
def dimension():
    pow2 = torch.from_numpy(
            np.array(
                [2**i for i in range(12)]
            )
        ).float()
    tmp = torch.Tensor(100,12)
    print(pow2.shape)
    res = torch.matmul(
        tmp,pow2
    )
    print(res.shape)
dimension()