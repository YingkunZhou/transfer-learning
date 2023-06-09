import argparse
import torch
import torch.nn as nn
import model as efficientformerv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.activations import *

def main(args):
    num_classes = args.num_classes
    weights = args.weights
    factor = args.factor
    convertion = args.convertion
    labels = args.label_path

    name = "efficientformerv2_" + factor
    create_model = getattr(efficientformerv2, name)

    # just use CPU to convert
    device = torch.device("cpu")
    activation = args.activation
    act_layer = nn.GELU
    if activation == 'relu':
        act_layer = nn.ReLU
    elif activation == 'nn.hardswish':
        act_layer = nn.Hardswish
    elif activation == 'hardswish':
        act_layer = HardSwish
    model = create_model(num_classes=num_classes, act_layer=act_layer).to(device)
    weights_dict = torch.load(weights, map_location=device)
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    for k in list(weights_dict.keys()):
        if "dist_head" in k:
            del weights_dict[k]
    model.load_state_dict(weights_dict)
    model.eval()

    size = 224
    inputs = torch.randn((1, 3, size, size))
    if convertion == 'all' or convertion == 'onnx':
        # https://pytorch.org/docs/stable/onnx.html
        # https://deci.ai/blog/how-to-convert-a-pytorch-model-to-onnx/
        torch.onnx.export(model, inputs, "onnx/%s-%d.onnx" % (name, num_classes), export_params=True, input_names=['input'], output_names=['output'])
        if convertion == 'onnx':
            return
    if convertion == 'all' or convertion == 'pd':
        # https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_with_x2paddle_cn.html
        from x2paddle.convert import pytorch2paddle
        pytorch2paddle(module=model,
                       save_dir="./pd_model",
                       jit_type="trace",
                       input_examples=[inputs],
                       enable_code_optim=False,
                       convert_to_lite=True,
                       lite_valid_places="arm",
                       lite_model_type="naive_buffer")
        if convertion == 'pd':
            return

    traced_model = torch.jit.trace(model, inputs)
    # followed by https://github.com/Tencent/ncnn/tree/master/tools/pnnx
    if convertion == 'all' or convertion == 'pt':
        traced_model.save("torchscript/%s-%d.pt" % (name, num_classes))
        if convertion == 'pt':
            return

    if convertion == 'all' or convertion == 'dqpt':
        model_dynamic_quantized = torch.quantization.quantize_dynamic(
            traced_model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        model_dynamic_quantized.save("torchscript/qint8-%s-%d.pt" % (name, num_classes))
        if convertion == 'dqpt':
            return

    if convertion == 'all' or convertion == 'sqpt':
        backend = "qnnpack"
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        model_static_quantized = torch.quantization.prepare(traced_model, inplace=False)
        model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
        model_static_quantized.save("torchscript/qnn-%s-%d.pt" % (name, num_classes))
        if convertion == 'sqpt':
            return

    if convertion == 'all' or convertion == 'mpt':
        from torch.utils.mobile_optimizer import optimize_for_mobile
        mobile_traced_model = optimize_for_mobile(traced_model)
        mobile_traced_model._save_for_lite_interpreter("torchscript/%s-%d.ptl" % (name, num_classes))
        if convertion == 'mpt':
            return

    import coremltools as ct
    if convertion == 'all' or convertion == 'coreml' or \
       convertion == 'package' or convertion == 'mlmodel':
        scale = len(IMAGENET_DEFAULT_STD) / sum(IMAGENET_DEFAULT_STD)
        bias = [-i/j for i, j in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]

        image_input = ct.ImageType(name = "input",
                                   shape=inputs.shape,
                                   scale=scale/255,
                                   bias=bias)

    if convertion == 'all' or convertion == 'coreml' or convertion == 'package':
        mlmodel = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[image_input],
            classifier_config=ct.ClassifierConfig(labels)
        )
        mlmodel.save("models/%s-%d.mlpackage" % (name, num_classes))

    if convertion == 'all' or convertion == 'coreml' or convertion == 'mlmodel':
        mlmodel = ct.convert(
            traced_model,
            inputs=[image_input],
            classifier_config=ct.ClassifierConfig(labels)
        )
        mlmodel.save("models/%s-%d.mlmodel" % (name, num_classes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--factor', type=str, default='s0')
    parser.add_argument('--weights', type=str, default="./weights/s0.best_model-gelu.pth")
    parser.add_argument('--activation', type=str, default="gelu")
    parser.add_argument('--label_path', type=str, default="../labels/flowers.txt")
    parser.add_argument('--convertion', type=str, default='all')

    opt = parser.parse_args()
    main(opt)
