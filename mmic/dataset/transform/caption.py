import torch
import numpy as np
import torch.nn.functional as F
def caption_collate_fn(data):
    sentences = [i[0] for i in data]
    labels = [i[1] for i in data]
    labels = torch.Tensor(np.array(labels)).long()
    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    cap_lengths = [count_trailing_zeros(cap) for cap in sentences]
    targets = torch.zeros(len(sentences),sentences[0][1].item()).long()
    # print(f'sentences {sentences}')
    for i, cap in enumerate(sentences):
        targets[i] = cap[0]

    cap_lengths = torch.Tensor(cap_lengths).long()
    return targets,labels,cap_lengths

def count_trailing_zeros(text):
    # 从后往前遍历tensor，直到遇到非零元素  
    for i in range(len(text) - 1, -1, -1):  
        if text[i] != 0:  
            # 遇到非零元素，返回已经计数的0的数量  
            return i+1
    # 如果整个tensor都是0，返回tensor的长度  
    return len(text)

def image_to_caption_collate_fn(data):
    """Build mini-batch tensors from a list of (image, sentence) tuples.
    Args:
      data: list of (image, sentence) tuple.
        - image: torch tensor of shape (3, 256, 256) or (?, 3, 256, 256).
        - sentence: torch tensor of shape (?); variable length.

    Returns:
      images: torch tensor of shape (batch_size, 3, 256, 256) or
              (batch_size, padded_length, 3, 256, 256).
      targets: torch tensor of shape (batch_size, padded_length).
      lengths: list; valid length for each padded sentence.
    """
    # Sort a data list by sentence length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, sentences, captions, ann_ids, image_ids, index = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    cap_lengths = [len(cap) for cap in sentences]
    targets = torch.zeros(len(sentences), max(cap_lengths)).long()
    for i, cap in enumerate(sentences):
        end = cap_lengths[i]
        targets[i, :end] = cap[:end]

    cap_lengths = torch.Tensor(cap_lengths).long()
    # print('cap_lengths', type(cap_lengths))
    return images, targets, captions, cap_lengths, ann_ids, image_ids, index