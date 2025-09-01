import copy
import random
import time
import torch
from clients.scaffold import ClientSCAFFOLD
from servers.serverbase import Server
import numpy as np

class SCAFFOLD(Server):
    def __init__(self, args):
        super().__init__(args)

        # select slow clients
        self.set_clients(ClientSCAFFOLD)
        self.set_late_clients(ClientSCAFFOLD)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        algorithmsettings = args['fedAlgorithm'][self.algorithm]
        self.server_learning_rate = algorithmsettings['server_learning_rate']
        self.fine_tuning_epoch = args['fine_tuning_epoch']
        self.global_c = []
        for param in self.global_model.parameters():
            self.global_c.append(torch.zeros_like(param))
    
    def test_generalized(self,new_attend_clients):
        generalized_accs,best_accs,self_accs = [],[],[]
        for lated_client in new_attend_clients:
            original_clients = np.random.choice(self.clients[:int(self.num_clients*(1-self.new_clients_rate))],5)
            original_avg_acc,original_test_num,original_auc = 0.0,0,0.0
            best_accuracy = 0.0
            for origin_client in original_clients:
                acc,num,auc = origin_client.test_other_model(lated_client.model)
                original_avg_acc += acc
                original_test_num += num
                original_auc += (auc * num)
                best_accuracy = max(best_accuracy,(acc*100.0)/num)
            original_avg_acc,original_auc = original_avg_acc / original_test_num,original_auc / original_test_num
            generalized_accs.append(original_avg_acc*100.0)
            best_accs.append(best_accuracy)
            
            self.slog.info('average test accuracy:{:.3f}%,avg_auc:{:.3f},and best accuracy:{:.3f}'.format(original_avg_acc*100.0,original_auc,best_accuracy))
            self.slog.info('-------finished generalization {} test--------'.format(lated_client.id))
            self.slog.info('starting to evaluate self performance')
            lated_accuracy,test_num,auc = lated_client.test_metrics()
            
            self.slog.info('self accuracy:{:.3f}%,auc:{:.3f}'.format((lated_accuracy*100.0)/test_num,auc))
            self_accs.append((lated_accuracy*100.0)/test_num)
        self.slog.info('generalized_accs:{},best_accs:{},self_accs:{}'.format(generalized_accs,best_accs,self_accs))
        
            
    def train(self):
        new_attend_clients = []
        self.slog.debug('server','starting to train')
        is_new_joined = False
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            if is_new_joined:
                self.test_generalized(new_attend_clients)
                
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.new_clients_settings['enabled'] and i >= self.start_new_joining_round and len(self.late_clients) > 0:
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
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
    
    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model, self.global_c)

            client.send_time['rounds'] += 1
            client.send_time['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        # self.delta_ys = []
        # self.delta_cs = []
        for client in active_clients:
            try:
                client_time_cost = client.train_time['total_cost'] / client.train_time['rounds'] + \
                        client.send_time['total_cost'] / client.send_time['rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.serial_id)
                self.uploaded_weights.append(client.train_samples)
                # self.delta_ys.append(client.delta_y)
                # self.delta_cs.append(client.delta_c)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        global_model = copy.deepcopy(self.global_model)
        global_c = copy.deepcopy(self.global_c)
        for cid in self.uploaded_ids:
            dy, dc = self.clients[cid].delta_yc()
            for server_param, client_param in zip(global_model.parameters(), dy):
                server_param.data += client_param.data.clone() / self.num_join_clients * self.server_learning_rate
            for server_param, client_param in zip(global_c, dc):
                server_param.data += client_param.data.clone() / self.num_clients
        self.global_model = global_model
        self.global_c = global_c
        
    # fine-tuning on new clients
    def fine_tuning_new_clients(self,new_attend_clients):
        for client in new_attend_clients:
            client.set_parameters(self.global_model, self.global_c)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for _ in range(self.fine_tuning_epoch):
                for _, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    
    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct,ns,auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
        
        ids = [c.id for c in self.clients]
        
        return ids,num_samples,tot_correct,tot_auc
    
    def evaluate(self,acc=None,loss=None):
        test_metrics_res = self.test_metrics()
        train_metrics_res = self.train_metrics()
        
        test_acc = sum(test_metrics_res[2]) * 1.0 / sum(test_metrics_res[1])
        test_auc = sum(test_metrics_res[3]) * 1.0 / sum(test_metrics_res[1])
        
        train_loss = sum(train_metrics_res[2]) * 1.0 / sum(train_metrics_res[1])
        accuracies = [correct / num for correct,num in zip(test_metrics_res[2],test_metrics_res[1])]
        #about auc, reference:https://zhuanlan.zhihu.com/p/569006692
        auc_collections = [acc / num for acc,num in zip(test_metrics_res[3],test_metrics_res[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)
        self.slog.info('server: avg train loss:{:.3f}'.format(train_loss))
        self.slog.info('server: avg test accuracy:{:.3f}'.format(test_acc))
        self.slog.info('server: avg test AUC:{:.3f}'.format(test_auc))
        
        self.slog.info('std: test accuracy:{:.3f}'.format(np.std(accuracies)))
        self.slog.info('std test AUC:{:.3f}'.format(np.std(auc_collections)))