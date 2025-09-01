import os
from .collector.multimodal import seprate_noniid

def generate():
    if not os.path.exists('repository/flickr30k'):
        print('you need to download by yourself, and put it under repository/flickr30k/')
        return
    seprate_noniid(purpose_type=0)
    seprate_noniid(purpose_type=1)
    return