EncodeDimension = {
    'densenet201':1000,
    'Mobile_net': 1280
}


MODALITIES = (
    'vision',
    'language',
    'Audio',
    # '',
)

DatasetModalities = {
    'flickr30k':['vision','text'],
    'crisis_mmd':['vision','text'],
    'food101':['vision','text']
}

def GetDatasetTestIndex(dataset,index):
    if dataset == 'flickr30k':
        return 'cluster'
    elif dataset == 'crisis_mmd':
        return 'cluster'
    else:
        return index