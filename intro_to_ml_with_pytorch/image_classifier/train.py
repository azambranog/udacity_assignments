from argparse import ArgumentParser as AP
from data_loaders import load_data
from make_network import FlowerNetwork
from train_functions import train_flower_model, calculate_test_accuracy
from manage_checkpoint import save_checkpoint
from workspace_utils import active_session


parser = AP(description='Train a NN to classify images using knowledge transfer')
parser.add_argument('data_directory')
parser.add_argument('--save_dir', dest="save_dir", default='.', help='Directory to save model, default is current directory')
parser.add_argument('--arch', dest="arch", default='vgg13', choices=['vgg11', 'vgg13', 'vgg16'], help='Architechture to train, default is vgg13')
parser.add_argument('--learning_rate', dest="learning_rate", default="0.01", type=float, help='Default is 0.01')
parser.add_argument('--hidden_units', dest="hidden_units", default=[256], type=int, nargs='*', help='Number of units of each layer. Default is one layer with 256 units')
parser.add_argument('--epochs', dest="epochs", default="5", type=int, help='Default is 5')
parser.add_argument('--dropout_rate', dest="dropout_rate", default="0.03", type=float, help='Default is 0.03')
parser.add_argument('--gpu', dest="gpu", action='store_true', help='Model should be trained on GPU')

args = parser.parse_args()

args.hidden_units = [int(u) for u in args.hidden_units]

print('#### Loading your data')
train_load, valid_load, test_load, class_to_idx = load_data(args.data_directory)

print('#### Creating base model')
model = FlowerNetwork(args.arch, args.hidden_units, drop_p=args.dropout_rate)


if args.gpu == True:
    dev = 'cuda'
else:
    dev = 'cpu'
print(f'#### Training model will start on {dev.upper()}')        
with active_session():
    model, optimizer = train_flower_model(model, train_load, class_to_idx, valid_load, epochs=args.epochs, 
                                      learn_rate=args.learning_rate, dropout_rate=args.dropout_rate, 
                                      dev=dev)

print('#### Calculating accuracy on test set')        
calculate_test_accuracy(model, test_load)

print('#### Saving model')
save_checkpoint(model, args.save_dir, args.arch, args.dropout_rate)
    
