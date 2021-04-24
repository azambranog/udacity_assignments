from argparse import ArgumentParser as AP
from manage_checkpoint import load_checkpoint
from predict_functions import predict

parser = AP(description='Classify an image of a flower')
parser.add_argument('input')
parser.add_argument('checkpoint')
parser.add_argument('--top_k', dest="top_k", default="3", type=int, help='Number of most probale classes to display. Default is 3')
parser.add_argument('--category_names', dest="category_names", help='Mapping JSON dictionary to class labels')
parser.add_argument('--gpu', dest="gpu", action='store_true', help='Model should be trained on GPU')

args = parser.parse_args()

print('##### Loading checkpoint')
model, chp_data = load_checkpoint(args.checkpoint)

if args.gpu == True:
    device = 'cuda'
else:
    device = 'cpu'

print(f'##### Predicting on {device.upper()}')
classes, ps = predict(args.input, model, chp_data, dev=device, topk=args.top_k, json_map=args.category_names)

print('{:=>40}'.format(''))
print('{:=^40}'.format('PREDICTIONS'))
print('{:=>40}'.format(''))
for i, c, p in zip(range(len(ps)), classes, ps):
    print(f'{i+1:<5}{c:.<25}{p.item()*100:.3f}%')

