from .client import ClientBase
from sklearn.preprocessing import label_binarize
import copy
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from models.optimizer.ditto import PersonalizedGradientDescent

class Ditto(ClientBase):
   def __init__(self,args,id,train_samples,test_samples,serial_id,logkey,**kwargs):
      super().__init__(args,id,train_samples,test_samples,serial_id,logkey,**kwargs)
      ditto = args['fedAlgorithm']['Ditto']
      self.mu = ditto['mu']
      # self.per_local_steps = ditto['per_local_steps']
      self.model_person = copy.deepcopy(self.model)
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
   '''
   具体来说,model.eval()的作用包括：

关闭dropout：在训练过程中,dropout是一种正则化技术,用于防止过拟合。
在评估模型时,我们希望使用全部的神经元（即关闭dropout）,以便得到更准确的结果。
关闭batch normalization层的训练模式：在训练期间,batch normalization层会在每个训练批次中计算其统计数据（均值和方差）。
然而,在测试或评估期间,我们希望使用在整个数据集上计算的统计数据,而不是每个测试批次单独计算。
因此,在评估模式下,batch normalization层会使用预先计算的统计数据,而不是在每个测试批次中实时计算。
禁用梯度计算：在训练过程中,我们通过反向传播算法计算梯度,并使用这些梯度来更新模型的权重。
在评估模式下,我们不会进行梯度计算,因此不会更新模型的权重。
   '''
       