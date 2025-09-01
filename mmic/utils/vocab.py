import pickle
import torch

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def load_from_pickle(self, data_path):
        with open(data_path, "rb") as fin:
            data = pickle.load(fin)
        self.idx = data["idx"]
        self.word2idx = data["word2idx"]
        self.idx2word = data["idx2word"]

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
    
    
def get_lengths(a)->list:
    result = []
    for row in a:
        indices = torch.where(row == 49407)[0]
            # 如果找到匹配的值，获取第一个匹配的索引
        first_occurrence = indices[0].item() if indices.numel() > 0 else len(a)
        if first_occurrence == -1:
            print(a)
        result.append(first_occurrence)
    return result