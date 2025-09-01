from threading import Thread
import torch.nn.functional as F
import utils.dlg as dlg
from algorithm.sim.lsh import SignRandomProjections
from models.optimizer.ditto import PersonalizedGradientDescent
from clients.ofchp import OFCHPClient
import time
from utils.data import read_client_data
from sklearn.cluster import KMeans
from cluster.clusterbase import ClusterBase
import torch
import const.constants as const
from algorithm.augmentation import augmentation
from .multimodalserver import Server
from clients.multimodal.pmcmfl import PmcmFLClient
import numpy as np
import os

class PmcmFLServer(Server):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.set_clients(PmcmFLClient)
        self.rep_dim = 64
        if self.dataset == 'food101':
            self.rep_dim = 512
    
    def train(self):
        self.slog.debug('server','starting to train')
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.validate_interface()

            for client in self.selected_clients:
                client.train()
            
            # Do the personalized client check
            if i % self.eval_gap == 0:
                self.attend_clients_validate()

            self.receive_models()
            self.aggregate_parameters()
            # self.check_global_model()
            self.budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.budget[-1])
            self.aggregate_local_prototypes(self.selected_clients,i)
    
    def aggregate_local_prototypes(self,clients,round):
        global_prototypes = {}
        img_sample_per_party_client = []
        text_sample_per_party_client = []
        fusion_sample_per_party_client = []
        target_sample_per_party_client = []
        for client in clients:
            img_sample_nums = 1 + client.sample_nums_with_modal['completed'] + client.sample_nums_with_modal["modal1only"]
            text_sample_nums = 1 + client.sample_nums_with_modal["completed"] + client.sample_nums_with_modal["modal2only"]
            fusion_sample_nums = 1 + client.sample_nums_with_modal["completed"]
            target_sample_nums = 1 + client.sample_nums_with_modal["completed"]
            img_sample_per_party_client.append(float(img_sample_nums))
            text_sample_per_party_client.append(float(text_sample_nums))
            fusion_sample_per_party_client.append(float(fusion_sample_nums))
            target_sample_per_party_client.append(float(target_sample_nums))
        img_proto_aggre_weight = [i / sum(img_sample_per_party_client) for i in img_sample_per_party_client]
        text_proto_aggre_weight = [i / sum(text_sample_per_party_client) for i in text_sample_per_party_client]
        fusion_proto_aggre_weight = [i / sum(fusion_sample_per_party_client) for i in fusion_sample_per_party_client]
        target_proto_aggre_weight = [i / sum(target_sample_per_party_client) for i in target_sample_per_party_client]

        all_client_prototypes_box = []
        all_client_prototype_path = []
        for client in clients:
            all_client_prototype_path.append(os.path.join(self.dataset_dir, 'client{}_prototypes-{}.pth'.format(client.serial_id,self.logkey) ))
        for prototype_path in all_client_prototype_path:
            all_client_prototypes_box.append(torch.load(prototype_path))
        for idx, prototypes in enumerate(all_client_prototypes_box):
            if idx == 0:
                global_prototypes["img"] = prototypes["img"] * img_proto_aggre_weight[idx]
                global_prototypes["text"] = prototypes["text"] * text_proto_aggre_weight[idx]
                global_prototypes["fusion"] = prototypes["fusion"] * fusion_proto_aggre_weight[idx]
                global_prototypes["target"] = prototypes["target"] * target_proto_aggre_weight[idx]
            else:
                global_prototypes["img"] += prototypes["img"] * img_proto_aggre_weight[idx]
                global_prototypes["text"] += prototypes["text"] * text_proto_aggre_weight[idx]
                global_prototypes["fusion"] += prototypes["fusion"] * fusion_proto_aggre_weight[idx]
                global_prototypes["target"] += prototypes["target"] * target_proto_aggre_weight[idx]
        zero_proto = torch.zeros([1, self.rep_dim], device="cpu", dtype=torch.float32)
        zero_target = torch.zeros([1, self.num_classes], device="cpu", dtype=torch.float32)
        total_img_weight_per_class = [0.0 for _ in range(self.num_classes)]
        total_text_weight_per_class = [0.0 for _ in range(self.num_classes)]
        total_fusion_weight_per_class = [0.0 for _ in range(self.num_classes)]
        total_target_weight_per_class = [0.0 for _ in range(self.num_classes)]
        
        for idx, prototypes in enumerate(all_client_prototypes_box):
            for clas in range(self.num_classes):
                if not torch.equal(prototypes["img"][clas:clas+1, :], zero_proto):
                    total_img_weight_per_class[clas] += img_proto_aggre_weight[idx]
                if not torch.equal(prototypes["text"][clas:clas+1, :], zero_proto):
                    total_text_weight_per_class[clas] += text_proto_aggre_weight[idx]
                if not torch.equal(prototypes["fusion"][clas:clas+1, :], zero_proto):
                    total_fusion_weight_per_class[clas] += fusion_proto_aggre_weight[idx]
                if not torch.equal(prototypes["target"][clas:clas+1, :], zero_target):
                    total_target_weight_per_class[clas] += target_proto_aggre_weight[idx]
        for clas in range(self.num_classes):
            if np.abs(total_img_weight_per_class[clas]) > 0.0001:
                global_prototypes["img"][clas:clas+1, :] /= total_img_weight_per_class[clas]
            if np.abs(total_text_weight_per_class[clas]) > 0.0001:
                global_prototypes["text"][clas:clas+1, :] /= total_text_weight_per_class[clas]
            if np.abs(total_fusion_weight_per_class[clas]) > 0.0001:
                global_prototypes["fusion"][clas:clas+1, :] /= total_fusion_weight_per_class[clas]
            if np.abs(total_target_weight_per_class[clas]) > 0.0001:
                global_prototypes["target"][clas:clas+1, :] /= total_target_weight_per_class[clas]

            # save global prototypes
        torch.save(global_prototypes, os.path.join(self.dataset_dir, 'global_prototypes-{}-{}.pth'.format(self.logkey,round)))