from .retrieval import RetrievalEvaluator

DatasetEvaluator = {
    'flickr30k':RetrievalEvaluator,
}

if __name__ == '__main__':
    obj = DatasetEvaluator['flickr30k']
    a = obj(None,'12','flickr30k')