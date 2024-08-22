import argparse
import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

from CaffeLoader import ModelParallel, loadCaffemodel
from losses import ContentLoss, StyleLoss, TVLoss
from utils import print_torch, setup_backends

Image.MAX_IMAGE_PIXELS = 1000000000  # Support gigapixel images


def add_options(parser: argparse.ArgumentParser):
    # Basic options
    parser.add_argument(
        "-image_size",
        help="Maximum height / width of generated image",
        type=int,
        default=512,
    )
    parser.add_argument(
        "-gpu",
        help="Zero-indexed ID of the GPU to use; for CPU mode set -gpu = c",
        default=0,
    )

    # Optimization options
    parser.add_argument("-content_weight", type=float, default=5e0)
    parser.add_argument("-style_weight", type=float, default=1e2)
    parser.add_argument("-normalize_weights", action="store_true")
    parser.add_argument("-normalize_gradients", action="store_true")
    parser.add_argument("-tv_weight", type=float, default=1e-3)
    parser.add_argument("-num_iterations", type=int, default=1000)
    parser.add_argument("-init", choices=["random", "image"], default="random")
    parser.add_argument("-init_image", default=None)
    parser.add_argument("-optimizer", choices=["lbfgs", "adam"], default="lbfgs")
    parser.add_argument("-learning_rate", type=float, default=1e0)
    parser.add_argument("-lbfgs_num_correction", type=int, default=100)

    # Output options
    parser.add_argument("-print_iter", type=int, default=50)
    parser.add_argument("-save_iter", type=int, default=100)
    parser.add_argument("-output_image", default="out.png")

    # Other options
    parser.add_argument("-style_scale", type=float, default=1.0)
    parser.add_argument("-original_colors", type=int, choices=[0, 1], default=0)
    parser.add_argument("-pooling", choices=["avg", "max"], default="max")
    parser.add_argument("-model_file", type=str, default="models/vgg19-d01eb7cb.pth")
    parser.add_argument("-disable_check", action="store_true")
    parser.add_argument(
        "-backend",
        choices=["nn", "cudnn", "mkl", "mkldnn", "openmp", "mkl,cudnn", "cudnn,mkl"],
        default="nn",
    )
    parser.add_argument("-cudnn_autotune", action="store_true")
    parser.add_argument("-seed", type=int, default=-1)

    parser.add_argument("-content_layers", help="layers for content", default="relu4_2")
    parser.add_argument(
        "-style_layers",
        help="layers for style",
        default="relu1_1,relu2_1,relu3_1,relu4_1,relu5_1",
    )

    parser.add_argument("-multidevice_strategy", default="4,7,29")
    return parser


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Input options
    parser.add_argument(
        "-style_image",
        help="Style target image",
        default="examples/inputs/seated-nude.jpg",
    )
    parser.add_argument("-style_blend_weights", default=None)
    parser.add_argument(
        "-content_image",
        help="Content target image",
        default="examples/inputs/tubingen.jpg",
    )
    parser = add_options(parser)
    params = parser.parse_args()
    return params


# Preprocess an image before passing it to a model.
# We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
# and subtract the mean pixel.
def preprocess(
    image_path: str, image_size: Union[int, tuple[int, int]]
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    if isinstance(image_size, int):
        image_size = tuple(
            [
                int((float(image_size) / max(image.size)) * x)
                for x in (image.height, image.width)
            ]
        )
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[[2, 1, 0]])])
    Normalize = transforms.Compose(
        [transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1, 1, 1])]
    )
    tensor = Normalize(rgb2bgr(Loader(image) * 255)).unsqueeze(0)
    return tensor


#  Undo the above preprocessing.
def deprocess(output_tensor: torch.Tensor) -> Image.Image:
    Normalize = transforms.Compose(
        [transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1, 1, 1])]
    )
    bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[[2, 1, 0]])])
    output_tensor = bgr2rgb(Normalize(output_tensor.squeeze(0).cpu())) / 255
    output_tensor.clamp_(0, 1)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor)
    return image


def prepare_inputs(
    content_image: str,
    style_image: str,
    image_size: int,
    style_scale: float,
    init_image: Optional[str],
    style_blend_weights: Optional[str],
    device: str,
) -> tuple[torch.Tensor, list[torch.Tensor], Optional[torch.Tensor], list[float]]:
    content_image = preprocess(content_image, image_size).to(device)
    assert content_image.dtype == torch.float

    style_image_input = style_image.split(",")
    style_image_list, ext = [], [".jpg", ".jpeg", ".png", ".tiff"]
    for image in style_image_input:
        if os.path.isdir(image):
            images = (
                image + "/" + file
                for file in os.listdir(image)
                if os.path.splitext(file)[1].lower() in ext
            )
            style_image_list.extend(images)
        else:
            style_image_list.append(image)
    style_images_caffe = []
    for image in style_image_list:
        style_size = int(image_size * style_scale)
        img_caffe = preprocess(image, style_size).to(device)
        assert img_caffe.dtype == torch.float
        style_images_caffe.append(img_caffe)

    if init_image is not None:
        image_size = (content_image.size(2), content_image.size(3))
        init_image = preprocess(init_image, image_size).to(device)
    else:
        init_image = None

    # Handle style blending weights for multiple style inputs
    if style_blend_weights is None:
        # Style blending not specified, so use equal weighting
        style_blend_weights_ = [1.0] * len(style_image_list)

        # for i in style_image_list:
        #     style_blend_weights.append(1.0)
        # for i, blend_weights in enumerate(style_blend_weights):
        #     style_blend_weights[i] = int(style_blend_weights[i])
    else:
        style_blend_weights_ = list(map(float, style_blend_weights.split(",")))
        assert (
            len(style_blend_weights_) == len(style_image_list)
        ), "-style_blend_weights and -style_images must have the same number of elements!"

    # Normalize the style blending weights so they sum to 1
    style_blend_sum = sum(style_blend_weights_)
    # for i, blend_weights in enumerate(style_blend_weights):
    #     style_blend_weights[i] = float(style_blend_weights[i])
    #     style_blend_sum = float(style_blend_sum) + style_blend_weights[i]
    style_blend_weights_ = [
        blend_weight / style_blend_sum for blend_weight in style_blend_weights_
    ]

    return (
        content_image,
        style_images_caffe,
        init_image,
        style_blend_weights_,
    )


def setup_multi_device(
    model: nn.Sequential, devices: list[str], multidevice_strategy: str
):
    device_splits = multidevice_strategy.split(",")
    assert (
        len(devices) - 1 == len(device_splits)
    ), "The number of -multidevice_strategy layer indices minus 1, must be equal to the number of -gpu devices."

    new_model = ModelParallel(model, devices, device_splits)
    return new_model


def setup_model(
    model_file: str,
    pooling: str,
    devices: list[str],
    multidevice_strategy: str,
    disable_check: bool = False,
    content_layers: str = "relu4_2",
    style_layers: str = "relu1_1,relu2_1,relu3_1,relu4_1,relu5_1",
    tv_weight: float = 1e-3,
    content_weight: float = 5e0,
    style_weight: float = 1e2,
    normalize_gradients: bool = False,
) -> tuple[
    Union[nn.Sequential, ModelParallel],
    list[ContentLoss],
    list[StyleLoss],
    list[TVLoss],
]:
    # load model
    model, model_info = loadCaffemodel(model_file, pooling, disable_check)

    content_layers = content_layers.split(",")
    style_layers = style_layers.split(",")

    # Set up the network, inserting style and content loss modules
    # model = copy.deepcopy(model)
    content_losses, style_losses, tv_losses = [], [], []
    next_content_idx, next_style_idx = 1, 1
    new_model = nn.Sequential()
    c, r = 0, 0
    if tv_weight > 0:
        tv_mod = TVLoss(tv_weight)
        new_model.add_module(str(len(new_model)), tv_mod)
        tv_losses.append(tv_mod)

    for i, layer in enumerate(list(model), 1):
        if next_content_idx <= len(content_layers) or next_style_idx <= len(
            style_layers
        ):
            if isinstance(layer, nn.Conv2d):
                new_model.add_module(str(len(new_model)), layer)

                if model_info["C"][c] in content_layers:
                    print(
                        "Setting up content layer "
                        + str(i)
                        + ": "
                        + str(model_info["C"][c])
                    )
                    loss_module = ContentLoss(content_weight, normalize_gradients)
                    new_model.add_module(str(len(new_model)), loss_module)
                    content_losses.append(loss_module)

                if model_info["C"][c] in style_layers:
                    print(
                        "Setting up style layer "
                        + str(i)
                        + ": "
                        + str(model_info["C"][c])
                    )
                    loss_module = StyleLoss(style_weight, normalize_gradients)
                    new_model.add_module(str(len(new_model)), loss_module)
                    style_losses.append(loss_module)
                c += 1

            if isinstance(layer, nn.ReLU):
                new_model.add_module(str(len(new_model)), layer)

                if model_info["R"][r] in content_layers:
                    print(
                        "Setting up content layer "
                        + str(i)
                        + ": "
                        + str(model_info["R"][r])
                    )
                    loss_module = ContentLoss(content_weight, normalize_gradients)
                    new_model.add_module(str(len(new_model)), loss_module)
                    content_losses.append(loss_module)
                    next_content_idx += 1

                if model_info["R"][r] in style_layers:
                    print(
                        "Setting up style layer "
                        + str(i)
                        + ": "
                        + str(model_info["R"][r])
                    )
                    loss_module = StyleLoss(style_weight, normalize_gradients)
                    new_model.add_module(str(len(new_model)), loss_module)
                    style_losses.append(loss_module)
                    next_style_idx += 1
                r += 1

            if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                new_model.add_module(str(len(new_model)), layer)

    if len(devices) > 1:
        new_model = setup_multi_device(new_model, devices, multidevice_strategy)
    else:
        new_model.to(devices[0])

    return new_model, content_losses, style_losses, tv_losses


# Configure the optimizer
def setup_optimizer(
    img: nn.Parameter,
    optimizer: str,
    learning_rate: float = 1e0,
    num_iterations: int = 1000,
    lbfgs_num_correction: int = 100,
) -> tuple[optim.Optimizer, int]:
    if optimizer == "lbfgs":
        print("Running optimization with L-BFGS")
        optim_state = {
            "max_iter": num_iterations,
            "tolerance_change": -1,
            "tolerance_grad": -1,
        }
        if lbfgs_num_correction != 100:
            optim_state["history_size"] = lbfgs_num_correction
        optimizer = optim.LBFGS([img], **optim_state)
        num_iterations = 1
    elif optimizer == "adam":
        print("Running optimization with ADAM")
        optimizer = optim.Adam([img], lr=learning_rate)
        num_iterations = num_iterations - 1
    else:
        raise NotImplementedError(f"Unsupported optimizer, {optimizer}.")
    return optimizer, num_iterations


# Combine the Y channel of the generated image and the UV/CbCr channels of the
# content image to perform color-independent style transfer.
def retain_original_colors(content: Image.Image, generated: Image.Image):
    content_channels = list(content.convert("YCbCr").split())
    generated_channels = list(generated.convert("YCbCr").split())
    content_channels[0] = generated_channels[0]
    return Image.merge("YCbCr", content_channels).convert("RGB")


# Divide weights by channel size
def normalize_weights(content_losses: list[ContentLoss], style_losses: list[StyleLoss]):
    for n, i in enumerate(content_losses):
        i.strength = i.strength / max(i.target.size())
    for n, i in enumerate(style_losses):
        i.strength = i.strength / max(i.target.size())


def main(params):
    # setup device and backend
    devices = params.gpu.split(",")
    devices = [
        ("cpu" if device.lower() == "c" else f"cuda:{device}") for device in devices
    ]
    backward_device = devices[0]
    setup_backends(
        devices, backend=params.backend, cudnn_autotune=params.cudnn_autotune
    )

    # preprocess inputs
    (
        content_image,
        style_images_caffe,
        init_image,
        style_blend_weights,
    ) = prepare_inputs(
        params.content_image,
        params.style_image,
        params.image_size,
        params.style_scale,
        params.init_image,
        params.style_blend_weights,
        backward_device,
    )

    model, content_losses, style_losses, tv_losses = setup_model(
        params.model_file,
        params.pooling,
        devices,
        params.multidevice_strategy,
        disable_check=params.disable_check,
        content_layers=params.content_layers,
        style_layers=params.style_layers,
        tv_weight=params.tv_weight,
        content_weight=params.content_weight,
        style_weight=params.style_weight,
        normalize_gradients=params.normalize_gradients,
    )
    # Freeze the network in order to prevent
    # unnecessary gradient calculations
    model.requires_grad_(False)
    # for param in model.parameters():
    #     param.requires_grad = False

    # Capture content targets
    for i in content_losses:
        i.mode = "capture"

    print("Capturing content targets")
    if len(devices) == 1:
        print_torch(model)
    with torch.no_grad():
        model(content_image)

    # Capture style targets
    for i in content_losses:
        i.mode = "None"

    for i, image in enumerate(style_images_caffe):
        print("Capturing style target " + str(i + 1))
        for j in style_losses:
            j.mode = "capture"
            j.blend_weight = style_blend_weights[i]
        with torch.no_grad():
            model(image)

    # Set all loss modules to loss mode
    for i in content_losses:
        i.mode = "loss"
    for i in style_losses:
        i.mode = "loss"

    # Maybe normalize content and style weights
    if params.normalize_weights:
        normalize_weights(content_losses, style_losses)

    # Initialize the image
    if params.seed >= 0:
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)
        torch.backends.cudnn.deterministic = True
    if params.init == "random":
        img = torch.randn_like(content_image).mul(0.001)
    elif params.init == "image":
        if init_image is not None:
            img = init_image.clone()
        else:
            img = content_image.clone()
    else:
        raise NotImplementedError(f"Unsupported init method, {params.init}.")
    img = nn.Parameter(img)

    def maybe_print(t, loss):
        if params.print_iter > 0 and t % params.print_iter == 0:
            print("Iteration " + str(t) + " / " + str(params.num_iterations))
            for i, loss_module in enumerate(content_losses):
                print(
                    "  Content " + str(i + 1) + " loss: " + str(loss_module.loss.item())
                )
            for i, loss_module in enumerate(style_losses):
                print(
                    "  Style " + str(i + 1) + " loss: " + str(loss_module.loss.item())
                )
            print("  Total loss: " + str(loss.item()))

    def maybe_save(t):
        should_save = params.save_iter > 0 and t % params.save_iter == 0
        should_save = should_save or t == params.num_iterations
        if should_save:
            output_filename, file_extension = os.path.splitext(params.output_image)
            if t == params.num_iterations:
                filename = output_filename + str(file_extension)
            else:
                filename = str(output_filename) + "_" + str(t) + str(file_extension)
            disp = deprocess(img.detach().clone())

            # Maybe perform postprocessing for color-independent style transfer
            if params.original_colors == 1:
                disp = retain_original_colors(deprocess(content_image.clone()), disp)

            disp.save(str(filename))

    # Function to evaluate loss and gradient. We run the net forward and
    # backward to get the gradient, and sum up losses from the loss modules.
    # optim.lbfgs internally handles iteration and calls this function many
    # times, so we manually count the number of iterations to handle printing
    # and saving intermediate results.
    num_calls = 0

    def feval():
        optimizer.zero_grad()
        model(img)
        loss = 0

        for mod in content_losses:
            loss += mod.loss.to(backward_device)
        for mod in style_losses:
            loss += mod.loss.to(backward_device)
        for mod in tv_losses:
            loss += mod.loss.to(backward_device)

        loss.backward()

        nonlocal num_calls
        num_calls += 1
        maybe_save(num_calls)
        maybe_print(num_calls, loss)

        return loss

    optimizer, num_iterations = setup_optimizer(
        img,
        params.optimizer,
        learning_rate=params.learning_rate,
        num_iterations=params.num_iterations,
        lbfgs_num_correction=params.lbfgs_num_correction,
    )
    while num_calls <= num_iterations:
        optimizer.step(feval)


if __name__ == "__main__":
    params = parse_args()
    main(params)
