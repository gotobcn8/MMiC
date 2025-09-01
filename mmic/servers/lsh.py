import copy
import numpy as np
import time
from clients.ditto import Ditto as ClientDitto
from .serverbase import Server
from .ditto import Ditto
from threading import Thread
import torch.nn.functional as F
import utils.dlg as dlg
from algorithm.sim.lsh import SignRandomProjections
from clients.lsh import LSHClient as ClientLshash
import time
from utils.data import read_client_data

class LSHServer(Server):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.fedAlgorithm = args['fedAlgorithm']['lsh']
        self.data_volume = self.fedAlgorithm['data_volume']
        self.hashF = SignRandomProjections(
            each_hash_num=self.fedAlgorithm['hash_num'],
            data_volume=self.data_volume,
            data_dimension=self.fedAlgorithm['cv_dim'],
            random_seed=args['random_seed']
        )
        self.sketches = dict()
        self.set_clients(ClientLshash)
    
    def set_for_lsh_clients(self,clientObj):
        self.set_clients(clientObj)
        #first we need to calculate the sketch for all clients.
        start_time = time.time()
        for client in self.clients:
            client.count_sketch(self.hashF)
            #  = self.hashF.hash(client)
            self.sketches[client.id] = client.minisketch
        
        self.slog.info('server :calculating time {:.3f}s'.format(time.time() - start_time))
    
    def set_late_clients(self,clientObj):
        for i in range(self.num_new_clients):
            train_data = read_client_data(self.dataset,i,self.dataset_dir,is_train=True)
            test_data = read_client_data(self.dataset,i,self.dataset_dir,is_train=False)
            client = clientObj(
                self.args,id = 'late_'+str(i),
                train_samples = len(train_data),
                test_samples = len(test_data),
            )
            self.late_clients.append(client)
            
    def train(self):
        self.slog.debug('server','starting to train')
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]
            
            self.receive_models()
            # if self.dlg_eval and i%self.dlg_gap == 0:
            #     self.call_dlg(i)
            self.aggregate_parameters()

            self.budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.budget[-1])

            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break
            if self.new_clients_settings['enabled'] and self.new_clients_joining_round >= i:
                if len(self.late_clients) < self.num_join_each_round:
                    new_attend_clients = self.late_clients
                else:
                    new_attend_clients = self.late_clients[:self.num_join_each_round]
                    self.late_clients = self.late_clients[self.num_join_each_round:]
                for new_client in new_attend_clients:
                    new_client.count_sketch()
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.budget[1:])/len(self.budget[1:]))

        self.save_results()
        self.save_global_model()

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientAVG)
        #     print(f"\n-------------Fine tuning round-------------")
        #     print("\nEvaluate new clients")
        #     self.evaluate()