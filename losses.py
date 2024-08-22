import torch
import torch.nn as nn


# Scale gradients in the backward pass
class ScaleGradients(torch.autograd.Function):
    @staticmethod
    def forward(self, input_tensor, strength):
        self.strength = strength
        return input_tensor

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input / (torch.norm(grad_input, keepdim=True) + 1e-8)
        return grad_input * self.strength * self.strength, None


# Define an nn Module to compute content loss
class ContentLoss(nn.Module):
    def __init__(self, strength, normalize):
        super(ContentLoss, self).__init__()
        self.strength = strength
        self.crit = nn.MSELoss()
        self.mode = "None"
        self.normalize = normalize

    def forward(self, input):
        if self.mode == "loss":
            loss = self.crit(input, self.target)
            if self.normalize:
                loss = ScaleGradients.apply(loss, self.strength)
            self.loss = loss * self.strength
        elif self.mode == "capture":
            self.target = input.detach()
        return input


class GramMatrix(nn.Module):
    def forward(self, input):
        B, C, H, W = input.size()
        x_flat = input.view(C, H * W)
        return torch.mm(x_flat, x_flat.t())


# Define an nn Module to compute style loss
class StyleLoss(nn.Module):
    def __init__(self, strength, normalize):
        super(StyleLoss, self).__init__()
        self.target = torch.Tensor()
        self.strength = strength
        self.gram = GramMatrix()
        self.crit = nn.MSELoss()
        self.mode = "None"
        self.blend_weight = None
        self.normalize = normalize

    def forward(self, input):
        self.G = self.gram(input)
        self.G = self.G.div(input.nelement())
        if self.mode == "capture":
            if self.blend_weight is None:
                self.target = self.G.detach()
            elif self.target.nelement() == 0:
                self.target = self.G.detach().mul(self.blend_weight)
            else:
                self.target = self.target.add(self.blend_weight, self.G.detach())
        elif self.mode == "loss":
            loss = self.crit(self.G, self.target)
            if self.normalize:
                loss = ScaleGradients.apply(loss, self.strength)
            self.loss = self.strength * loss
        return input


class TVLoss(nn.Module):
    def __init__(self, strength: float):
        super(TVLoss, self).__init__()
        self.strength = strength

    def forward(self, input: torch.Tensor):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.strength * (
            torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff))
        )
        return input
