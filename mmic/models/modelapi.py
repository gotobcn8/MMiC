import inspect
from enum import Enum
from typing import Callable,Dict
from .resnet import resnet
from .cnn import NormCNN,ImproveCNN
from .transformer import TransformerModel
from .retrieval import SCANpp
from .rnn import ImageTextClassifier, ProtoImageTextClassifier
from .multimodal import pretrain_clip, kancrisis


class ModelFactory:
    # model name -> method
    model_loaders: Dict[str, Callable[..., object]] = {
        'resnet': 'get_resnet_model',
        'resnet10': 'get_resnet_model',
        'resnet18': 'get_resnet_model',
        'resnet50': 'get_resnet_model',
        'clip':'get_clip_model',
        'protoclip':'get_protoclip_model',
        # 'clip': pretrain_clip.PartialPretrainedClip,
        'cnn': NormCNN,
        'cnn+': ImproveCNN,
        'transformer': TransformerModel,
        'scanpp': SCANpp,
        'NormTextImageClassifier': ImageTextClassifier,
        'ProtoTextImageClassifier':ProtoImageTextClassifier,
        'densenet': kancrisis.DenseNetBertMMModel,
    }

    @classmethod
    def get_model(cls, model_name: str,device, *args, **kwargs):
        """Model name to get the instance"""
        loader = cls.model_loaders.get(model_name)
        kwargs['device'] = device
        if not loader:
            raise ValueError(f"Model '{model_name}' is not registered.")
        # dynamic caller
        if isinstance(loader, str):  # method name
            return getattr(cls, loader)(model_name,device, *args, **kwargs)
        elif callable(loader):  # object
            kwargs = cls.get_valid_kwargs(loader,kwargs)
            return loader(*args, **kwargs).to(device)
        else:
            raise TypeError(f"Invalid loader type for model '{model_name}'.")
    
    @staticmethod
    def get_resnet_model(model_name: str,device,**kwargs:dict):
        """根据名称加载 ResNet 模型"""
        if model_name.endswith('18'):
            return resnet.resnet18_client(**kwargs).to(device)
        elif model_name.endswith('10'):
            return resnet.resnet10_client(**kwargs).to(device)
        elif model_name.endswith('50'):
            return resnet.resnet_50_client(**kwargs).to(device)
        else:
            raise ModuleNotFoundError(f"Couldn't find the {model_name}, please check it.")

    @staticmethod
    def get_clip_model(model_name:str,device:str, **kwargs:dict):
        model_aggregator = {}
        if kwargs.get('pretrained',True):
            # load pretrain clip
            model_aggregator['pretrained'] = pretrain_clip.get_pretrained_clip().to(device)
        if kwargs.get('partial',True):
            model_aggregator['partial'] = pretrain_clip.PartialPretrainedClip(kwargs['num_classes']).to(device)
            # model_aggregator['partial'] = pretrain_clip.GmcPartialClip(dataset_parameter['num_classes']).to(args["device"])
        return model_aggregator

    @staticmethod
    def get_protoclip_model(model_name:str,device:str, **kwargs:dict):
        model_aggregator = {}
        if kwargs.get('pretrained',True):
            # load pretrain clip
            model_aggregator['pretrained'] = pretrain_clip.get_pretrained_clip().to(device)
        if kwargs.get('partial',True):
            model_aggregator['partial'] = pretrain_clip.ProtoPretrainedClip(kwargs['num_classes']).to(device)
            # model_aggregator['partial'] = pretrain_clip.GmcPartialClip(dataset_parameter['num_classes']).to(args["device"])
        return model_aggregator
    
    @staticmethod
    def get_valid_kwargs(func, kwargs):
        # using inspect.signature extract function parameters
        sig = inspect.signature(func)
        valid_keys = sig.parameters.keys()
        # filter kwargs，only retain effective parameters
        return {k: v for k, v in kwargs.items() if k in valid_keys}
