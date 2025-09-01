from .serverbase import Server
from clients.fedavg import FedAvgClient
import time
import torch
import const.constants as const
import numpy as np

class FedAvg(Server):
    def __init__(self,args) -> None:
        super().__init__(args)
        self.set_clients(FedAvgClient)
        self.set_late_clients(FedAvgClient)
    

    def train(self):
        new_attend_clients = []
        is_new_joined = False
        self.slog.debug('server','starting to train')
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            if is_new_joined:
                self.evaluate_generalized(new_attend_clients)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.budget[-1])
            
            if self.new_clients_settings['enabled'] and i+1 >= self.start_new_joining_round and len(self.late_clients) > 0:
                if len(self.late_clients) < self.num_new_join_each_round:
                    new_attend_clients = self.late_clients
                    self.late_clients = []
                else:
                    new_attend_clients = self.late_clients[:self.num_new_join_each_round]
                    self.late_clients = self.late_clients[self.num_new_join_each_round:]
                #it need to be fine-tuned before attending
                self.fine_tuning_new_clients(new_attend_clients)
                self.clients.extend(new_attend_clients)
                is_new_joined = True
            else:
                is_new_joined = False
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.budget[1:])/len(self.budget[1:]))
        
        self.save_running_process()
        self.save_results()
        self.save_global_model()
    
    def fine_tuning_new_clients(self,new_clients):
        for new_client in new_clients:
            new_client.set_parameters(self.global_model)
            # new_client.model = copy.deepcopy(self.clusters[int(which_cluster)].cluster_model)
            optimizer = torch.optim.SGD(new_client.model.parameters(),lr = self.learning_rate)
            lossFunc = torch.nn.CrossEntropyLoss()
            train_loader = new_client.load_train_data()
            new_client.model.train()
            for _ in range(self.fine_tuning_epoch):
                for _,(x,y) in enumerate(train_loader):
                    if isinstance(x,list):
                        x[0] = x[0].to(new_client.device)
                        x = x[0]
                    else:
                        x = x.to(new_client.device)
                    y = y.to(new_client.device)
                    output = new_client.model(x)
                    loss =  lossFunc(output,y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
    
    def evaluate_generalized(self, new_attend_clients):
        generalized_accs,best_accs,self_accs = [],[],[]
        for late_client in new_attend_clients:
            be_test_clients = np.random.choice(
                a = self.clients,
                size = int(len(self.clients) * const.TEST_GENERALIZATION_RATE),
                replace = False,
            )
            original_avg_acc,original_test_num,original_auc = 0.0,0,0.0
            best_accuracy = 0.0
            for origin_client in be_test_clients:
                acc,num,auc = origin_client.test_other_model(late_client.model)
                original_avg_acc += acc
                original_test_num += num
                original_auc += (auc * num)
                best_accuracy = max(best_accuracy,(acc*100.0)/num)
            original_avg_acc,original_auc = original_avg_acc / original_test_num,original_auc / original_test_num
            generalized_accs.append(original_avg_acc*100.0)
            best_accs.append(best_accuracy)
            
            self.slog.info('average test accuracy:{:.3f}%,avg_auc:{:.3f},and best accuracy:{:.3f}'.format(original_avg_acc*100.0,original_auc,best_accuracy))
            self.slog.info('-------finished generalization {} test--------'.format(late_client.id))
            self.slog.info('starting to evaluate self performance')
            lated_accuracy,test_num,auc = late_client.test_metrics()
            
            self.slog.info('self accuracy:{:.3f}%,auc:{:.3f}'.format((lated_accuracy*100.0)/test_num,auc))
            self_accs.append((lated_accuracy*100.0)/test_num)
        self.slog.info('generalized_accs:{},best_accs:{},self_accs:{}'.format(generalized_accs,best_accs,self_accs))