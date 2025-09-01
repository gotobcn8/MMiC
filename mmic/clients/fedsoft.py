from .client import ClientBase
import torch
import numpy as np
import utils.data as data
from torch.utils.data import DataLoader
import copy
class FedSoftClient(ClientBase):
    def __init__(self,args,id,train_samples,test_samples,serial_id,logkey,**kwargs):
        super().__init__(args,id,train_samples,test_samples,serial_id,logkey,**kwargs)
        self.count_smoother = self.fedalgo['count_smoother']
        self.estimate_samples = self.fedalgo['estimate_samples']
        self.num_clusters = args['cluster']['cluster_num']
        
    def load_sample_data(self):
        estimate_data = data.read_client_data(self.dataset,self.serial_id,self.dataset_dir,is_train = True)
        return DataLoader(estimate_data,batch_size=self.estimate_samples, shuffle=True)
    
    def get_importance(self, count=True):
        if count:
            return [ust * self.train_samples for ust in self.importance_estimated]
        else:
            return self.importance_estimated
    
    def estimate_importance_weights(self,clusters):
         with torch.no_grad():
            table = np.zeros((self.num_clusters, self.estimate_samples))
            start_idx = 0
            nst_cluster_sample_count = [0] * self.num_clusters
            # sample_loader = DataLoader(self.ds, batch_size=256)
            sample_loader = self.load_sample_data()
            for x, y in sample_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                for s, cluster in enumerate(clusters):
                    cluster.cluster_model.eval()
                    # using cluster model to eval
                    out = cluster.cluster_model(x)
                    loss = self.loss(out, y)
                    # set the s-th cluster, from start_idx to start_idx+len(x) loss
                    table[s][start_idx:start_idx + len(x)] = loss.cpu()
                start_idx += len(x)

            # selecting the minimun loss cluster idx in each samples.
            min_loss_idx = np.argmin(table, axis=0)
            # statistic
            for s in range(self.num_clusters):
                # computing the times the highest the cluster it's.
                nst_cluster_sample_count[s] += np.sum(min_loss_idx == s)
            for s in range(self.num_clusters):
                if nst_cluster_sample_count[s] == 0:
                    nst_cluster_sample_count[s] = self.count_smoother * self.train_samples
            # get the minimun matrix for the cluster mini loss function.
            self.importance_estimated = np.array([1.0 * nst / self.train_samples for nst in nst_cluster_sample_count])
    
    def get_model_dict(self):
        return copy.deepcopy(self.model.state_dict())