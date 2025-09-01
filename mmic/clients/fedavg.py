from .client import ClientBase

class FedAvgClient(ClientBase):
    def __init__(self,args,id,train_samples,test_samples,serial_id,logkey,**kwargs):
        super().__init__(args,id,train_samples,test_samples,serial_id,logkey,**kwargs)