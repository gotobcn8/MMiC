from .clusterbase import EvalCluster
from .clusterbase import ClusterBase
import copy
import numpy as np
import const.constants as const
from fedlog.logbooker import glogger
class FedSoftCluster(EvalCluster):
    def __init__(self,args,id,model) -> None:
        super().__init__(args,id,model)
        # self.id = id
        # self.cluster_model = copy.deepcopy(model)
        
    def test_model_generalized(self, selected_clients,test_client):
        glogger.debug('Starting to evaluate the generalization of cluster {}'.format(self.id))
        avg_test_acc,total_test_num,avg_auc = 0,0,0
        if len(selected_clients) <= 1:
            glogger.warn('test_client:{}, cluster clients less than 1,can not test the generallization'.format(test_client.id))
            return
        if test_client.train_time['rounds'] == 0:
            glogger.info('this is first time to test the generalization of {}'.format(test_client.id))
        for sc in selected_clients:
            test_acc,test_num,auc = sc.test_other_model(test_client.model)
            glogger.info('cluster {} new coming client {} be tested client {}'.format(self.id,test_client.id,sc.id))
            glogger.info('test_accuracy:{:.3f}% test_num:{} correct_num:{} test_auc:{}'.format((test_acc*100.0)/test_num,test_num,test_acc,auc))
            total_test_num += test_num
            avg_test_acc += test_acc
            avg_auc += auc*test_num
        
        avg_test_acc =  (avg_test_acc * 1.0) / total_test_num 
        avg_auc =  (avg_auc * 1.0) / total_test_num
        glogger.info('{}-------avg_acc:{:.3f}% avg_auc:{:.3f}'.format(test_client.id,avg_test_acc*100,avg_auc))
    