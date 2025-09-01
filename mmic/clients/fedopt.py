from .client import ClientBase
from utils.multimodaldata.dataloader import load_multimodal_data

class FedOptCient(ClientBase):
    def __init__(self, args, id, train_samples, test_samples, serial_id, logkey, **kwargs):
        super().__init__(args, id, train_samples, test_samples, serial_id, logkey, **kwargs)
