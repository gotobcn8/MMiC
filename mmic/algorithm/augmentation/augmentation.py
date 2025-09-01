from .mixup import MixUp
from .mixup import ReMixup
from .rotate import Rotate
from utils.data import get_pure_client_data 
import numpy as np
import os

AugmentationMaps = {
    'mixup' : MixUp,
    'remixup':ReMixup,
    'rotate':Rotate,
}

def DoAugmentation(dataset_name,dataset_dir,client_sid,augment_type = 'mixup'):
    train_dataset = get_pure_client_data(dataset_name,client_sid,dataset_dir,is_train=True)
    if augment_type == 'mixup':
        x,y = AugmentationMaps[augment_type](
            dataset=train_dataset,
            mix_rate=1.0,
        )
    elif augment_type == 'remixup':
        x,y = AugmentationMaps[augment_type](
            dataset=train_dataset,
            mix_rate=1.0,
        )
    elif augment_type == 'rotate':
        x,y = AugmentationMaps[augment_type](
            dataset=train_dataset,
            mix_rate=1.0,
        )
    with open(os.path.join(dataset_dir,dataset_name,'train', str(client_sid) + '.npz'), 'wb') as f:
        np.savez(f, data = {'x':x,'y':y})
    return len(y)
