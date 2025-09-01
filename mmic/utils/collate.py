import torch

def pad_tensor(vec, pad):
    pad_size = list(vec.shape)
    pad_size[0] = pad - vec.size(0)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=0)

def collate_mm_fn_padd(batch):
    # find longest sequence
    if batch[0][0][0] is not None: max_a_len = max(map(lambda x: x[0][0].shape[0], batch))
    if batch[0][0][1] is not None: max_b_len = max(map(lambda x: x[0][1].shape[0], batch))
    
    # pad according to max_len
    x_a, x_b, len_a, len_b, ys = list(), list(), list(), list(), list()
    for idx in range(len(batch)):
        batch_x = batch[idx][0]
        x_a.append(pad_tensor(batch_x[0], pad=max_a_len))
        x_b.append(pad_tensor(batch_x[1], pad=max_b_len))
        
        len_a.append(batch_x[2])
        len_b.append(batch_x[3])
        # len_a.append(torch.tensor(batch[idx][0][2]))
        # len_b.append(torch.tensor(batch[idx][0][3]))

        ys.append(batch[idx][-1])
    
    # stack all
    x_a = torch.stack(x_a, dim=0)
    x_b = torch.stack(x_b, dim=0)
    # len_a = torch.stack(len_a, dim=0)
    len_a = torch.Tensor(len_a)
    len_b = torch.Tensor(len_b)
    # len_b = torch.stack(len_b, dim=0)
    ys = torch.stack(ys, dim=0)
    return [x_a, x_b, len_a, len_b], ys