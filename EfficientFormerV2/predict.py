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
from timm.layers.activations import *
import time


def cpu_timestamp(*args, **kwargs):
    # perf_counter returns time in seconds
    return time.perf_counter()


def main(args):
    num_classes = args.num_classes
    weights = args.weights
    json_path = args.json_path
    factor = args.factor

    #  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)

    img_size = 224
    # to make image preprocessing as same as coreml
    std = sum(IMAGENET_DEFAULT_STD) / len(IMAGENET_DEFAULT_STD)
    mean = [i / j * std for i, j in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
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
    img = img.to(device)

    # read class_indict
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    name = "efficientformerv2_" + factor
    create_model = getattr(efficientformerv2, name)
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
        if args.benchmark:
            time_min = 1e5
            time_avg = 0
            time_max = 0
            warmup_iterations = args.warmup_iter
            test_iterations = args.test_iter
            # warm-up
            for i in range(warmup_iterations):
                out = model(img)
            # TODO: every time load img again?
            for i in range(test_iterations):
                start_time = cpu_timestamp()
                out = model(img)
                end_time = cpu_timestamp()
                time = (end_time - start_time) * 1000.0
                time_min = time if time < time_min else time_min
                time_max = time if time > time_max else time_max
                time_avg += time
            time_avg /= test_iterations
            print("min = {:7.2f}  max = {:7.2f}  avg = {:7.2f}".format(time_min, time_max, time_avg))
            predict = torch.squeeze(out).cpu()
            for i in [985,723,872]:
                print("class: {:10}   prob: {:.11}".format(str(i), predict[i].numpy()))
            #  start_time = cpu_timestamp()
            #  for i in range(test_iterations):
                #  model(img)
            #  end_time = cpu_timestamp()
            #  print("Number of samples processed per second: {:.2f}".format(test_iterations / (end_time - start_time)))
        else:
            # predict class
            predict = torch.squeeze(model(img)).cpu()
            #  predict = torch.softmax(predict, dim=0)
            for i in range(len(predict)):
                print("class: {:10}   prob: {:.11}".format(class_indict[str(i)],
                                                       predict[i].numpy()))

    if not args.benchmark:
        predict_cla = torch.argmax(predict).numpy()
        plt.title("class: {}   prob: {:.11}".format(class_indict[str(predict_cla)],
                                                    predict[predict_cla].numpy()))
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--factor', type=str, default='s0')
    parser.add_argument('--weights', type=str, default="./weights/s0.best_model-gelu.pth")
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--json_path', type=str, default="../labels/flowers_indices.json")
    parser.add_argument('--benchmark', action=argparse.BooleanOptionalAction)
    # parser.add_argument('--benchmark', action='store_true') # for python 3.8
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--warmup_iter', type=int, default=20)
    parser.add_argument('--test_iter', type=int, default=100)

    opt = parser.parse_args()
    main(opt)
