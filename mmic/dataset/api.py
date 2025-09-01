
from .agnews import generate as generate_agnews
from .mnist import generate as generate_mnist
from .fmnist import generate as generate_fmnist
from .cifar10 import generate_augmentation as generate_cifar10
from .emnist import generate as generate_emnist
from .coco import generate as generate_coco
from .flickr30k import generate as generate_flickr30k

traditional_dataset_generator = {
    'agnews':generate_agnews,
    'mnist':generate_mnist,
    'fmnist':generate_fmnist,
    'cifar10':generate_cifar10,
    'emnist':generate_emnist,
}


def SpecialAlgorithm(args):
    algo_name = args['algorithm']
    
def creamfl_dataset_settings(args,algop):
    clients_num = algop['clients_num']
    dataset_settings = algop['dataset']
    for key,dataset in dataset_settings:
        if dataset in traditional_dataset_generator:
            traditional_dataset_generator[dataset](
                dir_path='repository',
                num_clients = clients_num[key],
                niid = True,
                balance=True,
                partition='dir',
            )
        