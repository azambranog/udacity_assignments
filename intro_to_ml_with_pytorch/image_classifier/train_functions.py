import torch
from torch import nn, optim
import time


def train_flower_model(model, train_load, class_to_idx, valid_load, epochs=5, learn_rate=0.003, dropout_rate=0.03, dev=None):
    model.class_to_idx = class_to_idx
    
    #choose device
    if dev is None:
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev_str = dev
        
    print('MODEL IS TRAINED IN {}'.format(dev_str.upper()))
    dev = torch.device(dev_str)
    
    #criterion is NLLLoss since the output is logsoftmax
    criterion = nn.NLLLoss()
    #we will only use Adam and not try other optimizers. Only the classifier part is to be trained
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    #moving the model to the right device
    model.to(dev)
    
    #Training
    step = 0
    checkpoint = 5
    running_loss = 0
    print_msgs=['']*10
    for e in range(epochs):
        # turn on dropout
        model.train()
        start = time.time()
        for images, labels in train_load:
            step += 1
            
            #move I/O to device
            images, labels = images.to(dev), labels.to(dev)
            
            #reset the gradient
            optimizer.zero_grad()
            
            logits = model.forward(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            #Validation
            if step % checkpoint == 0:
                # Turn dropout off
                model.eval()
                
                # Turn off gradients
                with torch.no_grad():
                    accuracy = 0
                    val_loss = 0
                    for images, labels in valid_load:
                        #move I/O to device
                        images, labels = images.to(dev), labels.to(dev)

                        logits = model.forward(images)
                        val_loss += criterion(logits, labels).item()

                        #Accuracy
                        ps = torch.exp(logits)
                        pred_is_right = (labels.data == ps.max(1)[1])

                        accuracy += pred_is_right.type_as(torch.FloatTensor()).mean()
                
                
                #nicely print everything, last ten lines, plus total plot
                train_loss = running_loss/checkpoint
                val_loss = val_loss/len(valid_load)
                val_acc = accuracy/len(valid_load)
                
                msg = "Epoch: {}/{}.. ".format(e+1, epochs) + "Batch: {}.. ".format(step) + \
                        "Training Loss: {:.3f}.. ".format(train_loss) + \
                        "Time per batch: {:.3f}.. ".format((time.time() - start)/checkpoint) + \
                        "Valid. Loss: {:.3f}.. ".format(val_loss)+ \
                        "Valid. Acc: {:.3f}".format(val_acc)
  
                print(msg)

                #reset stuff
                running_loss = 0
                start = time.time()
                # turn on dropout
                model.train()
    #return the model for use later
    return model, optimizer

def calculate_test_accuracy(model, test_load):
    if next(model.parameters()).is_cuda:
        dev = 'cuda'
    else:
        dev = 'cpu'
    
    #set to eval mode (turn dropout off)
    model.eval()

    # We don't want gradients
    with torch.no_grad():
        accuracy = 0
        for images, labels in test_load:
            #move I/O to device
            images, labels = images.to(dev), labels.to(dev)

            logits = model.forward(images)

            #Accuracy
            ps = torch.exp(logits)
            pred_is_right = (labels.data == ps.max(1)[1])
            accuracy += pred_is_right.type_as(torch.FloatTensor()).mean()
    
    accuracy= accuracy/len(test_load)
    print('The accuracy of the model in the test set is: {:.3f}%'.format(accuracy*100))
    
    return accuracy
    