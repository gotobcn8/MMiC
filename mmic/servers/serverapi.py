from .serverbase import Server
from .ditto import Ditto
from .lsh import LSHServer
from .ofchp import OFCHPServer
from .cfl import ClusterFL
from .serverbase import Server
from .ifca import IFCAServer
from .scaffold import SCAFFOLD
from .pacfl import PACFL
from .fedsoft import FedSoftServer
from .fedavg import FedAvg
from .creamfl import CreamFL
from .fedopt import FedOptServer

from .multimodal.fedavg import FedAvg as MMFedAvgServer
from .multimodal.mmic import MMiC
from .multimodal.fedopt import MMFedOptServer
from .multimodal.pacfl import PACFL as MMPACFLServer
from .multimodal.pmcmfl import PmcmFLServer
from fedlog.logbooker import glogger
from typing import Optional

ServersAPI = {
    'ditto':Ditto,
    'ofchp':OFCHPServer,
    'lsh':LSHServer,
    'cfl':ClusterFL,
    'fedavg':FedAvg,
    'ifca':IFCAServer,
    'scaffold':SCAFFOLD,
    'pacfl':PACFL,
    'fedsoft':FedSoftServer,
    'fedopt':FedOptServer,
}

MultimodalServersAPI = {
    'mmic':MMiC,
    'fedavg':MMFedAvgServer,
    'pacfl':MMPACFLServer,
    'fedopt':MMFedOptServer,
    'creamfl':CreamFL,
    'pmcmfl':PmcmFLServer,
}

def get_server(name:str, args:dict):
    name = name.lower()
    is_multimodal = args.get('is_multimodal',False)
    if is_multimodal and name in MultimodalServersAPI:
        return MultimodalServersAPI[name](args)
    elif not is_multimodal and name in ServersAPI:
        return ServersAPI[name](args)
    else:
        glogger.exception('please make sure your algorithm is supported in FedAdvan')