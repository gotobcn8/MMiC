from .encoder import StTextEncoder
from .transformer import PIENet
from .resnet.resnet import resnet18_client
from .pcme import PCME
import const.defaultp as dp

def get_creamfl_models(args):
    algo_name = args['algorithm']
    algop = args[algo_name]
    clients_model_maps = {}
    clients_model_maps['text'] = select_modelsby_name(args,'text',algop = algop) 
    clients_model_maps['image'] = select_modelsby_name(args,'image',algop = algop) 
    clients_model_maps['multimodal'] = select_modelsby_name(args,'multimodal',algop = algop) 

def select_modelsby_name(args,modal_name,algop):
    algo_dataset = algop['dataset'][modal_name]
    model_name = algop['models'][modal_name]
    if model_name  == 'encoderText':
        return StTextEncoder(
            word_dim = dp.ModelArgs['StTextEncoder']['word_dim'],
            # word_dim=200,
            embed_dim = dp.ModelArgs['StTextEncoder']['embedding_dim'],
            num_classes = algo_dataset['num_classes'],
        ).to(args["device"])
    elif model_name.startswith('resnet'):
        return resnet18_client(
                pretrained = dp.ModelArgs['resnet18']["pretrained"],
                num_classes = algo_dataset['num_classes'],
                is_train = True,
                scale = dp.ModelArgs['resnet18']["scale"],
                mlp_local = dp.ModelArgs['resnet18']["mlp_local"],
                embedding_dim = dp.ModelArgs['resnet18']["embedding_dim"],
            ).to(args["device"])
    elif model_name == 'pcme':
        return PCME(dp.ModelArgs['pcme']).to(args["device"])
