import os
import json
import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from timm.layers.activations import *

import model as mobilevit


def main(args):
    num_classes = args.num_classes
    weights = args.weights
    json_path = args.json_path
    factor = args.factor
    coreml_compatible = args.coreml_compatible

    #  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    img_size = 256
    data_transform = transforms.Compose(
        [transforms.Resize(img_size),
         transforms.CenterCrop(img_size),
         transforms.ToTensor()])

    # load image
    img_path = "../daisy.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    name = "mobile_vit_" + factor
    create_model = getattr(mobilevit, name)
    activation = args.activation
    act_layer = nn.SiLU
    if activation == 'relu':
        act_layer = nn.ReLU
    elif activation == 'nn.hardswish':
        act_layer = nn.Hardswish
    elif activation == 'hardswish':
        act_layer = HardSwish
    model = create_model(num_classes=args.num_classes, act_layer=act_layer, coreml_compatible=coreml_compatible).to(device)
    # load model weights
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        predict = torch.squeeze(model(img.to(device))).cpu()
        #  predict = torch.softmax(predict, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.11}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.11}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--factor', type=str, default='xx_small')
    parser.add_argument('--activation', type=str, default='silu')
    parser.add_argument('--weights', type=str, default="./weights/xxs.best_model-silu.pth")
    parser.add_argument('--json_path', type=str, default="../labels/flowers_indices.json")
    parser.add_argument('--coreml_compatible', action=argparse.BooleanOptionalAction)

    opt = parser.parse_args()
    main(opt)
