from .dataloader import read_multimodal_data
from .dataloader import read_retrieval_data

from .collate import RetrievalCollateFn
from .collate import ClassifierCollateFn
from .collate import RawClassifierCollate
DatasetGenerator = {
    'crisis_mmd': read_multimodal_data,
    'flickr30k':read_retrieval_data,
    'food101':read_multimodal_data,
}

CollateGenerator = {
    'flickr30k':RetrievalCollateFn,
    'crisis_mmd':ClassifierCollateFn,
    'food101':RawClassifierCollate,
}