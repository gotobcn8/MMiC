import numpy as np
import os
import sys
import random
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# from torchtext.vocab import set_default_index
from .collector.tools import separate_data, split_data, save_file, check
from .transform.caption import caption_collate_fn
from utils.vocab import Vocabulary
import pickle

random.seed(777)
np.random.seed(777)
num_clients = 20
num_classes = 4
max_len = 200
dir_path = "agnews/"
is_coco_vocab = True


def generate(dir_path, num_clients, num_classes, niid, balance, partition, alpha):
    if not os.path.isabs(dir_path):
        dir_path = os.path.join("repository", dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # Setup directory for train/test data
    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")
    raw_path = os.path.join(dir_path, "raw")

    if check(
        config_path,
        train_path,
        test_path,
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        alpha,
    ):
        return

    trainset, testset = torchtext.datasets.AG_NEWS(root=raw_path)
    trainlabel, traintext = list(zip(*trainset))
    testlabel, testtext = list(zip(*testset))

    dataset_text = []
    dataset_label = []

    dataset_text.extend(traintext)
    dataset_text.extend(testtext)
    dataset_label.extend(trainlabel)
    dataset_label.extend(testlabel)

    if is_coco_vocab:
        with open("repository/vocabs/coco_vocab.pkl", "rb") as fin:
            coco_vocab = pickle.load(fin)
        word2idx = coco_vocab["word2idx"]

        # 创建标记化器
        def coco_tokenizer(text):
            tokens = []
            for word in text.split():
                if word in word2idx:
                    tokens.append(word2idx[word])
                else:
                    tokens.append(word2idx["<unk>"])
            return tokens

        vocab = word2idx
        text_pipline = lambda x: coco_tokenizer(x)
        label_pipline = lambda x: int(x) - 1
    else:
        tokenizer = get_tokenizer("basic_english")
        vocab = build_vocab_from_iterator(
            map(tokenizer, iter(dataset_text)), specials=["<unk>"]
        )
        vocab.set_default_index(vocab["<unk>"])
        text_pipline = lambda x: vocab(tokenizer(x))
        label_pipline = lambda x: int(x) - 1

    def text_transform(text, label, max_len):
        label_list, text_list = [], []
        for _text, _label in zip(text, label):
            label_list.append(label_pipline(_label))
            text_ = text_pipline(_text)
            padding = [0 for i in range(max_len - len(text_))]
            text_.extend(padding)
            text_list.append(text_[:max_len])
        return label_list, text_list

    label_list, text_list = text_transform(dataset_text, dataset_label, max_len)
    text_lens = [len(text) for text in text_list]
    text_list = [(text, i) for text, i in zip(text_list, text_lens)]

    text_list = np.array(text_list, dtype=object)
    label_list = np.array(label_list, dtype=object)

    # starting to split the dataset
    X, y, statistic, overview = separate_data(
        (text_list, label_list),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        alpha,
    )

    train_data, test_data = split_data(X, y)
    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition,
        overview,
        alpha,
    )
    print("the size of vocabulary:", len(vocab))

if __name__ == "__main__":
    niid = True if sys.argv[1] == "noiid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate(dir_path, num_clients, num_classes, niid, balance, partition)
