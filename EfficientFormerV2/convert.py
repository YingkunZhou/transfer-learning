import sys
import coremltools as ct
import torch
import model as efficientformerv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

if __name__ == '__main__':
    num_classes = int(sys.argv[1])
    if num_classes == 5:
        weights = "weights/best_model-relu.pth"
        labels = "../labels/flowers.txt"
    else:
        weights = "weights/mobilevit_xxs.pt"
        labels = "../labels/imagenet-labels.txt"

    factor = sys.argv[2]
    name = "efficientformerv2_" + factor 

    create_model = getattr(efficientformerv2, name)

    device = torch.device("cpu")
    model = create_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
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
    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        classifier_config=ct.ClassifierConfig(labels)
    )
    mlmodel.save("models/%s-%d.mlmodel" % (name, num_classes))

    mlmodel = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[image_input],
        classifier_config=ct.ClassifierConfig(labels)
    )
    mlmodel.save("models/%s-%d.mlpackage" % (name, num_classes))
