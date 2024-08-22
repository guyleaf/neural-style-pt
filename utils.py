import torch
import torch.nn as nn


def setup_cuda(backend: str = "nn", cudnn_autotune: bool = False):
    if "cudnn" in backend:
        torch.backends.cudnn.enabled = True
        if cudnn_autotune:
            torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.enabled = False


def setup_cpu(backend: str = "nn"):
    if "mkl" in backend and "mkldnn" not in backend:
        torch.backends.mkl.enabled = True
    elif "mkldnn" in backend:
        raise ValueError("MKL-DNN is not supported yet.")
    elif "openmp" in backend:
        torch.backends.openmp.enabled = True


def setup_backends(
    devices: list[str], backend: str = "nn", cudnn_autotune: bool = False
):
    if "cpu" in devices:
        setup_cpu(backend)
    if any("cuda" in device for device in devices):
        setup_cuda(backend, cudnn_autotune)


# Print like Torch7/loadcaffe
def print_loadcaffe(model: nn.Sequential, model_info: dict[str, list[str]]):
    cnn_info = model_info["C"]

    i = 0
    for layer in model:
        if isinstance(layer, nn.Conv2d):
            print(
                f"{cnn_info[i]}: {layer.out_channels} {layer.in_channels} {layer.kernel_size}"
            )
            i += 1
        if i == len(cnn_info):
            break


# Print like Lua/Torch7
def print_torch(model: nn.Sequential):
    simplelist = ""
    for i, layer in enumerate(model, 1):
        simplelist = simplelist + "(" + str(i) + ") -> "
    print("nn.Sequential ( \n  [input -> " + simplelist + "output]")

    def strip(x):
        return str(x).replace(", ", ",").replace("(", "").replace(")", "") + ", "

    def n():
        return "  (" + str(i) + "): " + "nn." + str(layer).split("(", 1)[0]

    for i, layer in enumerate(model, 1):
        if "2d" in str(layer):
            ks, st, pd = (
                strip(layer.kernel_size),
                strip(layer.stride),
                strip(layer.padding),
            )
            if "Conv2d" in str(layer):
                ch = str(layer.in_channels) + " -> " + str(layer.out_channels)
                print(
                    n()
                    + "("
                    + ch
                    + ", "
                    + (ks).replace(",", "x", 1)
                    + st
                    + pd.replace(", ", ")")
                )
            elif "Pool2d" in str(layer):
                st = st.replace("  ", " ") + st.replace(", ", ")")
                print(
                    n() + "(" + ((ks).replace(",", "x" + ks, 1) + st).replace(", ", ",")
                )
        else:
            print(n())
    print(")")
