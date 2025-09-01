import torch

def test_reshape_functin():
    a = torch.Tensor(20,40)
    a = a.reshape(-1,20)
    b = torch.Tensor(23,20)
    assert a.shape == (40,20)
    assert b.shape == (23,20)
    c = torch.cat([a,b],dim = 0)
    assert c.shape == (63,20)


def test_tuple_index():
    a = (1,2,3,4,5,6)
    index = [2,3,4]
    b = a[index[:]]
    assert len(b) == 3