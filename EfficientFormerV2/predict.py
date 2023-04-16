import os
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import model as efficientformerv2


def main(args):
    num_classes = args.num_classes
    weights = args.weights
    json_path = args.json_path
    factor = args.factor

    #  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    img_size = 224
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
    name = "efficientformerv2_" + factor
    create_model = getattr(efficientformerv2, name)
    model = create_model(num_classes=num_classes).to(device)
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
    parser.add_argument('--factor', type=str, default='s0')
    parser.add_argument('--weights', type=str, default="./weights/best_model-gelu.pth")
    parser.add_argument('--json_path', type=str, default="../labels/flowers_indices.json")

    opt = parser.parse_args()
    main(opt)
