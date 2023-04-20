import torch
import torch.nn as nn
import model as mobilevit
from torchsummary import summary
import argparse

def main(args):
    size = args.size
    name = args.name
    activation = args.activation
    act_layer = nn.SiLU
    if activation == 'relu':
        act_layer = nn.ReLU
    elif activation == 'nn.hardswish':
        act_layer = nn.Hardswish
    elif activation == 'hardswish':
        act_layer = HardSwish
    elif activation == 'gelu':
        act_layer = nn.GELU

    create_model = getattr(mobilevit, 'mobile_vit_' + name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=1000, act_layer=act_layer).to(device)
    summary(model, (3,size,size))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--name', type=str, default='xx_small')
    parser.add_argument('--activation', type=str, default="silu")

    opt = parser.parse_args()
    main(opt)
