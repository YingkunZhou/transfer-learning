import sys
import coremltools as ct
import torch
import model as mobilevit

if __name__ == '__main__':
    num_classes = int(sys.argv[1])
    if num_classes == 5:
        weights = "weights/best_model-hardswish.pth"
        labels = "labels/flowers.txt"
    else:
        weights = "weights/mobilevit_xxs.pt"
        labels = "labels/imagenet-labels.txt"

    factor = ''
    if len(sys.argv) == 3:
        factor = sys.argv[2] + '_'
    name = "mobile_vit_%ssmall" % factor

    create_model = getattr(mobilevit, name)

    device = torch.device("cpu")
    model = create_model(num_classes=num_classes, coreml_compatible=True).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    size = 256
    inputs = torch.randn((1, 3, size, size))
    traced_model = torch.jit.trace(model, inputs)

    image_input = ct.ImageType(name="input", shape=inputs.shape, scale=1. / 255.)

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
