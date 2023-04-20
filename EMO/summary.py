import torch
import torch.nn as nn
import model as mobilevit
from torchsummary import summary
import argparse

def main(args):
    size = args.size
    name = args.name
    activation = args.activation

    create_model = getattr(mobilevit, 'EMO_' + name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=1000, act_layer=activation).to(device)
    summary(model, (3,size,size))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--name', type=str, default='1M')
    parser.add_argument('--activation', type=str, default="silu")

    opt = parser.parse_args()
    main(opt)
