import torch
from PIL import Image
import numpy as np
import json

def predict(image_path, model, checkpoint, dev='cpu', topk=5, json_map=None):
    train_index_translation = {str(v): int(k) for k, v in checkpoint['class_to_idx'].items()}
    
    
    top_p, top_class = predict_image(image_path, model, dev=dev, topk=topk)
    
    if dev == 'cuda':
        top_class = top_class.cpu()
        top_p = top_p.cpu()
        
    classes = top_class.numpy().reshape(-1)
    classes = [str(train_index_translation[str(c)]) for c in classes]
    if json_map is not None:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]
    
    ps = top_p.numpy().reshape(-1)
    
    return classes, ps
    

def predict_image(image_path, model, dev='cpu', topk=5):
    model.to(dev)
    
    img = torch.from_numpy(process_image(image_path)) 
    img.to(dev)
    if dev == 'cuda':
        img = img.type(torch.cuda.FloatTensor)
    else:
        img = img.float()
     
    img.unsqueeze_(0)
        
    with torch.no_grad():
        model.eval()
        logits = model(img)
    ps = torch.exp(logits)
    top_p, top_class = ps.topk(topk)
    return top_p, top_class

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    #resize
    h, w = img.size
    r = 256/min(h, w)
    img.thumbnail((r*h, r*w))
    #crop
    w, h = img.size
    l = (w - 224)/2
    r = l + 224
    t = (h - 224)/2
    b = t + 224
    img = img.crop((l, t, r, b))
    
    #nomrlize
    img = np.array(img)/255
    mu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mu) / std
    
    #reshape
    img = img.transpose((2, 0, 1))
    return img