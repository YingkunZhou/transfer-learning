import argparse
import coremltools as ct
import torch
import torch.nn as nn
import model as EMO
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.activations import *

def main(args):
    num_classes = args.num_classes
    activation = args.activation
    weights = args.weights
    factor = args.factor
    coreml = args.coreml
    labels = args.label_path

    name = "EMO_" + factor
    create_model = getattr(EMO, name)

    device = torch.device("cpu")
    model = create_model(num_classes=num_classes, act_layer=activation).to(device)
    weights_dict = torch.load(weights, map_location=device)
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    for k in list(weights_dict.keys()):
        if "dist_head" in k:
            del weights_dict[k]
    model.load_state_dict(weights_dict)
    model.eval()

    size = 224
    inputs = torch.randn((1, 3, size, size))
    traced_model = torch.jit.trace(model, inputs)

    scale = len(IMAGENET_DEFAULT_STD) / sum(IMAGENET_DEFAULT_STD)
    bias = [-i/j for i, j in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]

    image_input = ct.ImageType(name = "input",
                               shape=inputs.shape,
                               scale=scale/255,
                               bias=bias)

    if coreml == 'all' or coreml == 'model':
        mlmodel = ct.convert(
            traced_model,
            inputs=[image_input],
            classifier_config=ct.ClassifierConfig(labels)
        )
        mlmodel.save("models/%s-%d.mlmodel" % (name, num_classes))

    if coreml == 'all' or coreml == 'package':
        mlmodel = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[image_input],
            classifier_config=ct.ClassifierConfig(labels)
        )
        mlmodel.save("models/%s-%d.mlpackage" % (name, num_classes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--factor', type=str, default='1M')
    parser.add_argument('--weights', type=str, default="./weights/1M.best_model-gelu.pth")
    parser.add_argument('--activation', type=str, default="silu")
    parser.add_argument('--label_path', type=str, default="../labels/flowers.txt")
    parser.add_argument('--coreml', type=str, default='all')

    opt = parser.parse_args()
    main(opt)
