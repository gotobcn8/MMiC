


def get_pcme()->dict:
    return {
        "embedding_dim":256,
        "n_samples_inference":7,
        "not_bert":True,
        "cnn_type":'resnet18',
        "mlp_local":False,
        "word_dim":300,
    }

def get_scanpp()->dict:
    return {
        "embedding_dim":512,
    }

class SimilarityEncoder():
    def __init__(self) -> None:
        self.regulator = 'only_rar'
        self.rar_step = 2
        self.attn_type = 't2i'
        self.t2i_smooth = 10.0
        
StEncoderParameters = {
    'word_dim' : 300,
    'embedding_dim' : 256,
}        

Resnet = {
    'pretrained':False,
    'scale':128,
    'embedding_dim': 256,
    'parameters':True,
    
}

ModelArgs = {
    'pcme':get_pcme(),
    'scanpp': get_scanpp(),
    'SimilarityEncoder': SimilarityEncoder(),
    'StTextEncoder': StEncoderParameters,
    'resnet18':Resnet,
    
}

# if __name__ == '__main__':
#     a = scnpp()
#     print(a['regulator'])