from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import CLIPProcessor
from .processer import pretrain
from PIL import Image
# from transformers import CLIPProcessor, CLIPModel

class Food101Dataset(Dataset):
    def __init__(self,x_data,labels) -> None:
        super().__init__()
        self.images,self.captions = zip(*x_data)
        #  = x_data[1]
        self.labels = labels
        # self.processer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        # self.processor = pretrain.ClipProcessor
        # self.caption_tokenizer = 
        self.image_transform = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image,caption,label = self.images[index],self.captions[index],self.labels[index]
        image = Image.open(image)
        # inputs = self.processer(text=caption,images = image,return_tensors = 'pt', padding=True)
        # return (inputs['pixel_values'],inputs['input_ids']),label,index
        return (image,caption),label,index