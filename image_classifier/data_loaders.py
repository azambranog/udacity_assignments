from torchvision import datasets, transforms
import torch 

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    base_tranforms = [transforms.Resize(255),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                     ]

    train_trans = transforms.Compose([transforms.RandomRotation(30, expand=True),
                                      transforms.RandomAffine(0, translate=(.2, .2), shear=30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip()] + 
                                     base_tranforms) 

    test_val_trans = transforms.Compose(base_tranforms) 

    train_dat = datasets.ImageFolder(train_dir, transform=train_trans)
    valid_dat = datasets.ImageFolder(valid_dir, transform=test_val_trans)
    test_dat = datasets.ImageFolder(test_dir, transform=test_val_trans)


    train_load = torch.utils.data.DataLoader(train_dat, batch_size=64, shuffle=True)
    valid_load = torch.utils.data.DataLoader(valid_dat, batch_size=64)
    test_load = torch.utils.data.DataLoader(test_dat, batch_size=64)
    
    class_to_idx = train_dat.class_to_idx
    return train_load, valid_load, test_load, class_to_idx