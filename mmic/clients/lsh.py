from .client import ClientBase
from models.optimizer.ditto import PersonalizedGradientDescent
import copy
from algorithm.sim.lsh import ReflectSketch
import utils.data as data
import numpy as np
from torch.utils.data import DataLoader
import time

class LSHClient(ClientBase):
    def __init__(self,args,id,train_samples,test_samples,serial_id,logkey,**kwargs):
        super().__init__(args,id,train_samples,test_samples,serial_id,logkey,**kwargs)
        lshAlgo = args['fedAlgorithm']['lsh']
        # self.per_local_steps = ditto['per_local_steps']
        self.model = copy.deepcopy(args['model'])
        # self.optimizer_personl = PersonalizedGradientDescent(
        #         self.model_person.parameters(), lr=self.learning_rate, mu=self.mu)
    
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
            self.reflector.get_sketch(x,self.device)
        self.sketch = self.reflector.sketch
        self.minisketch = self.reflector.sketch / self.reflector.NumberData
        self.clog.info('{} :calculate sketch time {:.3f}s'.format(self.id,time.time()-start_time))
        return self.minisketch
        
    def load_sketch_data(self,data_volume = 1000):
        sketch_data = data.read_x_data(
            self.dataset,
            self.serial_id,
            self.dataset_dir,
            is_train = True
        )
        sketch_data = np.random.choice(sketch_data[0],data_volume)
        
        return DataLoader(
            dataset = sketch_data,
            batch_size = data_volume,
            shuffle = True
        )
        
        
if __name__ == '__main__':
    before = time.time()
    print(time.time() - before)