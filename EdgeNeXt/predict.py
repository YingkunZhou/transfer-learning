import os
import json
import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import model as efficientformerv2
from activations import HardSwish


def main(args):
    num_classes = args.num_classes
    weights = args.weights
    json_path = args.json_path
    factor = args.factor

    #  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    img_size = 256
    # to make image preprocessing as same as coreml
    std = sum(IMAGENET_DEFAULT_STD) / len(IMAGENET_DEFAULT_STD)
    mean = [i/j*std for i, j in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
    data_transform = transforms.Compose(
        [transforms.Resize(img_size),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])

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
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    name = "edgenext_" + factor
    create_model = getattr(edgenext, name)
    activation = args.activation
    act_layer = nn.GELU
    if activation == 'relu':
        act_layer = nn.ReLU
    elif activation == 'nn.hardswish':
        act_layer = nn.Hardswish
    elif activation == 'hardswish':
        act_layer = HardSwish
    model = create_model(num_classes=num_classes, act_layer=act_layer).to(device)
    # load model weights
    weights_dict = torch.load(weights, map_location=device)
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    for k in list(weights_dict.keys()):
        if "dist_head" in k:
            del weights_dict[k]
    model.load_state_dict(weights_dict)
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
    parser.add_argument('--weights', type=str, default="./weights/xxs.best_model-gelu.pth")
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--json_path', type=str, default="../labels/flowers_indices.json")

    opt = parser.parse_args()
    main(opt)
