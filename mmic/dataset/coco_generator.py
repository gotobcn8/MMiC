import pickle
import numpy as np
from utils.vocab import Vocabulary
import os
import random
from torch.utils.data import DataLoader
import torch
from utils.transform import imagenet_transform
from utils.transform import caption_transform
from .coco import CocoCaptionsCap
from utils.loader import image_to_caption_collate_fn

def load_vocab(vocab_path):
    if isinstance(vocab_path, str):
        vocab = Vocabulary()
        vocab.load_from_pickle(vocab_path)
    else:
        vocab = vocab_path
    return vocab



def _get_coco_file_paths(dataset_root):
    """Select proper train / val classes and omit id files."""
    train_ids = np.load(os.path.join(dataset_root,"annotations/coco_train_ids.npy"))
    train_extra_ids = np.load(os.path.join(dataset_root,"annotations/coco_restval_ids.npy"))
    val_ids = np.load(os.path.join(dataset_root,"annotations/coco_dev_ids.npy"))[:5000]
    te_ids = np.load(os.path.join(dataset_root,"annotations/coco_test_ids.npy"))

    image_root = os.path.join(dataset_root, "allimages")
    train_ann = os.path.join(dataset_root, "annotations/captions_train2014.json")
    val_ann = os.path.join(dataset_root, "annotations/captions_val2014.json")

    return train_ids, train_extra_ids, val_ids, te_ids, image_root, train_ann, val_ann


def _get_coco_loader(
    image_root,
    annotation_path,
    ids,
    vocab,
    num_workers,
    batch_size=64,
    train=False,
    extra_ids=None,
    extra_annotation_path=None,
    cutout_prob=0.0,
    caption_drop_prob=0.0,
    subset=False,
    subset_num=50000,
    client=-1,
):
    _image_transform = imagenet_transform(
        random_resize_crop=train,
        random_erasing_prob=cutout_prob,
    )
    _caption_transform = caption_transform(vocab, caption_drop_prob)

    coco_dataset = CocoCaptionsCap(
        image_root,
        annotation_path,
        extra_annFile=extra_annotation_path,
        ids=ids,
        extra_ids=extra_ids,
        transform=_image_transform,
        target_transform=_caption_transform,
        client=client,
    )

    if subset:
        if not os.path.exists("coco_subset_idx_file"):
            full_idx = [i for i in range(566435)]
            random.shuffle(full_idx)
            idx = full_idx[0:50000]
            idx.sort()
            if not os.path.exists("coco_subset_idx_file"):
                with open("coco_subset_idx_file", "wb") as f:
                    pickle.dump(idx, f)

        if subset_num == 50000:
            with open("coco_subset_idx_file", "rb") as f:
                idx = pickle.load(f)

        coco_dataset = torch.utils.data.Subset(coco_dataset, idx)

    elif client > -1:
        idx = [i for i in range(100000 + client * 10000, 110000 + client * 10000)]
        coco_dataset = torch.utils.data.Subset(coco_dataset, idx)

    dataloader = DataLoader(
        coco_dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=image_to_caption_collate_fn,
        pin_memory=True,
    )
    if subset or client > -1:
        print(f"Loading COCO Caption: n_captions {len(coco_dataset)}...")
    else:
        print(
            f"Loading COCO Caption: n_images {coco_dataset.n_images} n_captions {len(coco_dataset)}..."
        )
    return dataloader
