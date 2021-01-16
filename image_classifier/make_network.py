from torch import nn
import torch.nn.functional as F
from torchvision import models

#Helper Class to dynamically create the classifier
class FlowerClassifier(nn.Module):
    def __init__(self, hidden_layers_size, drop_p=0.03):
        super().__init__()
        
        #input layer, vgg16 features section has 25088 neurons output
        self.hidden = nn.ModuleList([nn.Linear(25088, hidden_layers_size[0])])
        
        #puplate the hidden layers
        layer_sizes = zip(hidden_layers_size[:-1], hidden_layers_size[1:])
        for i in range(len(hidden_layers_size) - 1):
            self.hidden.extend([nn.Linear(hidden_layers_size[i], hidden_layers_size[i+1])])
        
        #we have 102 flower categories
        self.output = nn.Linear(hidden_layers_size[-1], 102)
        
        #set dropout
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        for layer in self.hidden:
            x = self.dropout(F.relu(layer(x)))
        
        x = F.log_softmax(self.output(x), dim=1)
        
        return x

# function to create the model
# hidden layers will be an hyperparameter
def FlowerNetwork(basemodel, hidden_layers, drop_p=0.03):
    #freeze model parameters
    model = getattr(models, basemodel)(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    
    #replace the classifier by our custom floer classifier
    model.classifier = FlowerClassifier(hidden_layers, drop_p=drop_p)
    
    return model