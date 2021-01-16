import torch
from glob import glob
from os import path
import re
from make_network import FlowerNetwork

def save_checkpoint(model, out_dir, arch, dropout_rate):
    
    checkpoint = {'hidden_layers': [each.out_features for each in model.classifier.hidden],
              'class_to_idx':  model.class_to_idx,
              'state_dict': model.state_dict(),
              'dropout_rate': dropout_rate,
              'arch': arch
             }
    available_checkpoints = glob('checkpoint*.pth')     
    available_checkpoints = [re.sub(r'checkpoint(.*)\.pth$', '\\1', f) for f in available_checkpoints]
    available_checkpoints = [int(x) for x in available_checkpoints if is_int(x)]
    if len(available_checkpoints) == 0:
        n = '0'
    else:
        n = str(max(available_checkpoints) + 1)
    
    out_filename = path.join(out_dir, f'checkpoint{n}.pth')
    
    torch.save(checkpoint, out_filename)
    print(f'Model is vailable at: {out_filename}')

def load_checkpoint(checkpoint):
    data = torch.load(checkpoint)
    model = FlowerNetwork(data['arch'], data['hidden_layers'], drop_p=data['dropout_rate'])
    model.load_state_dict(data['state_dict'])
    return model, data
    
    
def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False