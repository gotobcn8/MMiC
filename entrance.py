# import models.models as models_sum
import mmic.utils.read as yml
import mmic.utils.cmd.parse as parse
import mmic.models.transformer as transformer
from mmic.models.cnn import NormCNN
from mmic.models.cnn import ImproveCNN
from mmic.models.rnn import ImageTextClassifier
# from models.cnn import ModerateCNN
from mmic.models.encoder import StTextEncoder
from mmic.models.multimodal.kancrisis import DenseNetBertMMModel 
from mmic.models.pcme import PCME
from mmic.models.retrieval import SCANpp
import mmic.const.defaultp as dp 
import functools

# from servers.serverapi import get_server
import mmic.servers.serverapi as sapi
import torch
from mmic.fedlog.logbooker import glogger
from mmic.dataset import download
import os
import mmic.const.supporter as support
from mmic.models.resnet.resnetclient import resnet18_client
# from models import api
from mmic.models.multimodal import pretrain_clip
from mmic.models.modelapi import ModelFactory
# tianzhen = 'tianzhenaa'
# logger.exception(f"couldn't find {tianzhen} in models")
import random
import numpy as np

def set_random_seed(seed: int = 42):
    random.seed(seed)              
    np.random.seed(seed)                
    torch.manual_seed(seed)               
    torch.cuda.manual_seed(seed)             
    torch.cuda.manual_seed_all(seed)      

def run(args):
    # cuda device
    
    if args["device"] == "gpu":
        if not torch.cuda.is_available():
            args["device"] = "cpu"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args["device_id"])
            # args["device"] = "cuda"
            args['device'] = torch.device(f'cuda:{args.get("device_id",0)}')
    # download.
    set_random_seed(666)
    glogger.info("Selecting models")
    # args["model"] = models_select(args)
    args['model'] = get_model(args)
    glogger.info("Selecting dataset")
    dataset_download(args)
    glogger.info("Selecting federated algorithm")
    server = sapi.get_server(args["algorithm"], args)
    server.train()

def dataset_download(args):
    # download the dataset
    dataset_name = args["dataset"]
    if dataset_name in support.MultimodalDatasets:
        download.multimodal_download(args, dataset_name)
    else:
        get_traditional_dataset(args, dataset_name)

def get_traditional_dataset(args, dataset):
    dtsparameter = args[dataset]
    download.download(
        name = dataset,
        niid = dtsparameter["niid"],
        balance = dtsparameter["balance"],
        partition = dtsparameter["partition"],
        num_clients = args["num_clients"],
        num_classes = dtsparameter["num_classes"],
        alpha = dtsparameter["alpha"],
    )
# def models_select(args):
#     model_name = args["model_name"]
#     algo_name = args['algorithm']
#     if algo_name in support.SpecialModel:
#         return api.ModelBricklayer(args, algo_name)
#     else:
#         return get_traditional_model(args, model_name, args["dataset"])

def get_model(args):
    model_name = args["model_name"]
    dataset_name = args['dataset']
    model_params = args['models'].get(model_name, dp.ModelArgs.get(model_name))
    model_params = model_params if model_params is not None else {}
    dataset_params = args[dataset_name]
    checkpoint_path = args.get('checkpoint')
    if checkpoint_path:
        return torch.load(checkpoint_path)
    args["batch_size"] = dataset_params.get('batch_size')
    # model_params['device'] = args['device']
    return ModelFactory.get_model(model_name=model_name,device = args['device'],**model_params,**dataset_params)

def get_traditional_model(args, model_name, dataset):
    parameters = args["models"].get(model_name,dp.ModelArgs.get(model_name))
    dataset_parameter = args[dataset]
    checkpoint_path = args.get('checkpoint')
    if checkpoint_path:
        return torch.load(checkpoint_path)
    if "batch_size" in parameters.keys():
        args["batch_size"] = parameters["batch_size"]
    if model_name.startswith("resnet"):
        if model_name == "resnet18":
            return resnet18_client(
                pretrained=parameters["pretrained"],
                num_classes=dataset_parameter["num_classes"],
                is_train=True,
                scale=parameters["scale"],
                mlp_local=parameters["mlp_local"],
                embedding_dim=parameters["embedding_dim"],
            ).to(args["device"])
    if model_name == "cnn":
        glogger.info("using cnn")
        # here should return the nn.Module
        if "mnist" in dataset:
            return NormCNN(
                in_features=1,
                num_classes=dataset_parameter["num_classes"],
                dim=parameters["dim"],
            ).to(args["device"])
        if dataset == "cifar10":
            return ImproveCNN(
                in_features=3,
                num_classes=dataset_parameter["num_classes"],
                dim=parameters["dim"],
            ).to(args["device"])
    elif model_name == "transformers":
        glogger.info("using transformer")
        return transformer.TransformerModel(
            ntoken=parameters["vocab_size"],
            d_model=parameters["embadding_dim"],
            nhead=8,
            d_hid = parameters["embadding_dim"],
            nlayers = parameters["nlayers"],
            num_classes = dataset_parameter["num_classes"],
        ).to(args["device"])
    elif model_name == "textencoder":
        return StTextEncoder(
            word_dim=parameters["word_dim"],
            # word_dim=200,
            embed_dim=parameters["embedding_dim"],
            num_classes=dataset_parameter["num_classes"],
        ).to(args["device"])
    elif model_name == "scanpp":
        if dataset == 'flickr30k':
            return SCANpp(parameters).to(args['device'])
    elif model_name == 'NormTextImageClassifier':
        return ImageTextClassifier(
            num_classes=dataset_parameter["num_classes"],
            img_input_dim = parameters['img_dim'],
            text_input_dim = parameters['text_dim'],
            missing_contrast=False,
        ).to(args["device"])
    elif model_name.lower() == 'clip':
        model_aggregator = {}
        if parameters.get('pretrained',True):
            # load pretrain clip
            model_aggregator['pretrained'] = pretrain_clip.get_pretrained_clip().to(args["device"])
        if parameters.get('partial',True):
            model_aggregator['partial'] = pretrain_clip.PartialPretrainedClip(dataset_parameter['num_classes']).to(args["device"])
            # model_aggregator['partial'] = pretrain_clip.GmcPartialClip(dataset_parameter['num_classes']).to(args["device"])
        return model_aggregator
    elif model_name.lower() == 'densenet':
        return DenseNetBertMMModel(
            dim_visual_repr = parameters['img_dim'],
            dim_text_repr = parameters['text_dim'],
            dim_proj = 100,
            num_class  = dataset_parameter["num_classes"],
        ).to(args["device"])
        
    glogger.exception(
        f"couldn't find {model_name} in models, plz check whether fedAdvan support it"
    )

def algorithm_select(algorithm: str, args):
    return 

def entrancelog(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Executing {func.__name__} with arguments {args} and {kwargs}")
        result = func(*args, **kwargs)
        return result
    return wrapper