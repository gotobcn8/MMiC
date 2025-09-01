from transformers import CLIPProcessor

ClipProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

LIMIT_TOKEN_SIZE = 77

def after_clipprocessor(data):
    '''
    clip only support images and texts
    '''
    images,texts = data
    res = ClipProcessor(text=texts,images=images,return_tensors='pt',padding=True)
    input_ids,attention_mask = res['input_ids'],res['attention_mask']
    if input_ids.size(1) > LIMIT_TOKEN_SIZE:
        res['input_ids'] = input_ids[:,:LIMIT_TOKEN_SIZE]
        res['attention_mask'] = attention_mask[:,:LIMIT_TOKEN_SIZE]
    return res