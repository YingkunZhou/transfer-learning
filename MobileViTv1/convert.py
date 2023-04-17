import argparse
import coremltools as ct
import torch
import torch.nn as nn
import model as mobilevit
from timm.layers.activations import *

def main(args):
    num_classes = args.num_classes
    weights = args.weights
    factor = args.factor
    coreml = args.coreml
    labels = args.label_path
    coreml_compatible = args.coreml_compatible

    name = "mobile_vit_" + factor
    create_model = getattr(mobilevit, name)

    device = torch.device("cpu")
    activation = args.activation
    act_layer = nn.SiLU
    if activation == 'relu':
        act_layer = nn.ReLU
    elif activation == 'nn.hardswish':
        act_layer = nn.Hardswish
    elif activation == 'hardswish':
        act_layer = HardSwish
    model = create_model(num_classes=args.num_classes, act_layer=act_layer, coreml_compatible=coreml_compatible).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    size = 256
    inputs = torch.randn((1, 3, size, size))
    traced_model = torch.jit.trace(model, inputs)

    image_input = ct.ImageType(name="input", shape=inputs.shape, scale=1. / 255.)

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
    parser.add_argument('--factor', type=str, default='xx_small')
    parser.add_argument('--activation', type=str, default='silu')
    parser.add_argument('--weights', type=str, default="./weights/xxs.best_model-silu.pth")
    parser.add_argument('--label_path', type=str, default="../labels/flowers.txt")
    parser.add_argument('--coreml', type=str, default='all')
    parser.add_argument('--coreml_compatible', action=argparse.BooleanOptionalAction)

    opt = parser.parse_args()
    main(opt)
