import utils.data as data
import numpy as np
from torch.utils.data import DataLoader
import time
from .client import ClientBase
from models.optimizer.ditto import PersonalizedGradientDescent
import copy
from algorithm.sim.lsh import ReflectSketch
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
from algorithm.augmentation import mixup
import const.constants as const
class OFCHPClient(ClientBase):
    def __init__(self,args,id,train_samples,test_samples,serial_id,logkey,**kwargs):
        super().__init__(args,id,train_samples,test_samples,serial_id,logkey,**kwargs)
        lshAlgo = args['fedAlgorithm'][self.algorithm]
        self.mu = lshAlgo['mu']
        self.model_person = copy.deepcopy(args['model'])
        self.optimizer_personl = PersonalizedGradientDescent(
                self.model_person.parameters(), lr=self.learning_rate, mu=self.mu)


    def train(self):
        train_loader = self.load_train_data()
       
        start_time = time.time()
        max_local_epochs = self.local_epochs
        self.model.train()
        for step in range(max_local_epochs):
            for i,(x,y) in enumerate(train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output,y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time['rounds'] += 1
        self.train_time['total_cost'] += time.time() - start_time
      
    def train_personalized(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()
        max_local_epochs = self.local_epochs
        #switch on training mode
        self.model_person.train()
        for step in range(max_local_epochs):
            for x, y in trainloader:
                if isinstance(x,list):
                    x[0] = x[0].to(self.device)
                    x = x[0]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model_person(x)
                loss = self.loss(output, y)
                self.optimizer_personl.zero_grad()
                loss.backward()
                self.optimizer_personl.step(self.model.parameters(),self.device)
        self.train_time['total_cost'] += time.time() - start_time
                
    def test_metrics(self):
        return self.test_personalized_with_metrics()
   
    def test_personalized_with_metrics(self):
        testloaderfull = self.load_test_data()
      
        self.model_person.eval()
      
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x,y in testloaderfull:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                    x = x[0]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model_person(x)
                
                test_acc += (torch.sum(torch.argmax(output,dim=1) == y)).item()
                test_num += y.shape[0]
                
                y_prob.append(F.softmax(output,dim=1).detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(),classes = np.arange(self.num_classes)))
        
        y_prob = np.concatenate(y_prob,axis=0)
        y_true = np.concatenate(y_true,axis=0)
        
        auc = metrics.roc_auc_score(y_true,y_prob,average='micro')
        return test_acc,test_num,auc
    
    def train_personalized_with_metrics(self):
        trainloader = self.load_train_data()
        self.model_person.eval()
        
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x,y in trainloader:
                if isinstance(x,list):
                    x[0] = x[0].to(self.device)
                    x = x[0]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model_person(x)
                loss = self.loss(output,y)
                #tensor拼接
                gm = torch.cat([p.data.view(-1) for p in self.model.parameters()],dim = 0)
                pm = torch.cat([p.data.view(-1) for p in self.model_person.parameters()],dim = 0)
                
                loss += 0.5 * self.mu * torch.norm(gm-pm,p = 2)
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses,train_num
    
    def count_sketch(self,hashF):
        self.reflector = ReflectSketch(
            hashF=hashF,
            dtype=float,
            data_vol=hashF.data_rows,
            hash_num = hashF.hash_num,
            dimension=hashF.dimension,
        )
        start_time = time.time()
        sketch_data = self.load_sketch_data(hashF.data_rows)
        for x in sketch_data:
            # x = x.reshape(-1,1)
            self.reflector.get_sketch(x,self.device)
        self.sketch = self.reflector.sketch
        self.minisketch = self.reflector.sketch / self.reflector.NumberData
        # count_sketch_time = time.time() - start_time
        self.clog.info('{} :calculate sketch time {:.3f}s'.format(self.id,time.time()-start_time))
        return self.minisketch
        
    def load_sketch_data(self,data_volume = 1000):
        sketch_data = data.read_x_data(
            self.dataset,
            self.serial_id,
            self.dataset_dir,
            data_type = const.DataType[0]
        )
        selected_data = np.random.choice(sketch_data.shape[0],data_volume)
        selected_sketch_data = sketch_data[selected_data]
        # do more data augmentation
        # if sketch_data.shape[0] < data_volume:
            # sketch_data = torch.tensor(sketch_data)
            # augmentation_data = torch.rot90(sketch_data,k = 3,dims=(2,3))
            # selected_sketch_data = torch.cat((sketch_data,augmentation_data),dim = 0)
            # selected_sketch_data = selected_sketch_data.numpy()
            # selected_sketch_data = mixup.MixUp(sketch_data,mix_rate=1.0)
        # selected_sketch_data = torch.cat((selected_sketch_data,augmentation_data),dim = 0)
        return DataLoader(
            dataset = selected_sketch_data,
            batch_size = data_volume,
            shuffle = True
        )