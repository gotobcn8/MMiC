import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from PIL import Image
from utils.vocab import Vocabulary
from functools import partial
from torchvision import transforms
from utils.transform import caption_transform
from utils.transform import imagenet_transform
import torch.nn.functional as F
import const.constants as const

class MultimodalDataset(Dataset):
    def __init__(self, indices, total_data, image_path, tag="train"):
        self.indice = indices
        self.image_path = image_path
        self.data = np.array(total_data[tag])[self.indice]
        vocab_path = 'repository/vocabs/coco_vocab.pkl'
        vocab = Vocabulary()
        vocab.load_from_pickle(vocab_path)
        self.target_transform = caption_transform(vocab,0)
        self.image_transform = imagenet_transform(
            # random_resize_crop=True if tag == 'train' else False,
            random_erasing_prob=0.0
        )
    def __getitem__(self, index):
        data = self.data[index]
        caption = data[1]

        img_path = os.path.join(self.image_path, data[0])
        img_obj = Image.open(img_path).convert("RGB")
        img = self.image_transform(img_obj)
        target = self.target_transform(caption)
        return img, target, caption, index, int(index / 5), index

    def __len__(self):
        return len(self.data)


def read_data(dataset, dir, idx, data_type):
    """
    dir = absolute directory,
    dataset is dataset name
    suffix is train or test data name
    idx is the indicator
    """
    prefix = data_type
    file_name = str(idx) + const.SUFFIX_NPZ
    full_file_name = os.path.join(dir, dataset, prefix, file_name)
    # current_file_path = os.path.abspath(__file__)
    # print(current_file_path)
    with open(full_file_name, "rb") as f:
        data = np.load(f, allow_pickle=True)["data"].tolist()
    return data

def read_missing_data(dir,idx,missing_file = 'missing.pickle'):
    file_name = os.path.join(dir,missing_file)
    try:
        with open(file_name, "rb") as f:
            data = np.load(f, allow_pickle=True)
        return data.get(idx)
    except:
        print(f'Couldn\' load {file_name}, no this missing file')
    
    
def read_client_data(dataset, idx, dir, is_train):
    if dataset == "agnews":
        return read_client_data_text(dataset, idx, dir, is_train)

    if is_train:
        train_data = read_data(dataset, dir, idx, const.DataType[0])
        X_train = torch.Tensor(train_data["x"]).type(torch.float32)
        y_train = torch.Tensor(train_data["y"]).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, dir, idx, const.DataType[1])
        X_test = torch.Tensor(test_data["x"]).type(torch.float32)
        y_test = torch.Tensor(test_data["y"]).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def get_pure_client_data(dataset, idx, dir, is_train):
    if is_train:
        train_data = read_data(dataset, dir, idx, is_train)
        return train_data
    else:
        test_data = read_data(dataset, dir, idx, is_train)
        return test_data


def read_x_data(dataset, idx, dir, data_type):
    # if dataset == 'agnews':
    #     return read_client_data_text(dataset,idx,dir,is_train)

    if data_type == const.DataType[0]:
        train_data = read_data(dataset, dir, idx, data_type)
        # train_x = np.array(train_data['x'])
        train_x = train_data["x"]
        for i in range(len(train_x)):
            train_x[i] = torch.tensor(train_x[i])
        return train_x
    else:
        test_data = read_data(dataset, dir, idx, data_type)
        # X_test = torch.Tensor(test_data['x']).type(torch.float32)
        return test_data["x"]


def read_client_data_text(dataset, idx, dir, is_train):
    read_data_info = []
    if is_train:
        read_data_info = read_data(dataset, dir, idx, True)
    else:
        read_data_info = read_data(dataset, dir, idx, False)
    X_list, X_list_lens = list(zip(*read_data_info["x"]))
    y_list = read_data_info["y"]

    X_list = torch.Tensor(X_list).type(torch.int64)
    X_list_lens = torch.Tensor(X_list_lens).type(torch.int64)
    y_list = torch.Tensor(y_list.astype("int32")).type(torch.int64)

    predict_data = [((x, lens), y) for x, lens, y in zip(X_list, X_list_lens, y_list)]
    return predict_data


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def load_pkl(path):
    return pickle.load(open(path, "rb"))
