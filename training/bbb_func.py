import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import vmap, grad_and_value

###################### Layers #########################
def reparam_linear(input, weight_mean, weight_rho, bias_mean, bias_rho):
    batch_size = input.shape[0]
    out_features = weight_mean.shape[0]

    weight_std = F.softplus(weight_rho)
    bias_std = F.softplus(bias_rho)

    batch_in = torch.stack((input, input**2))
    batch_mat = torch.stack((weight_mean.transpose(0, 1), weight_std.transpose(0, 1)**2))
    batch_add = torch.stack((bias_mean.expand((batch_size, out_features)), (bias_std**2).expand((batch_size, out_features))))
    batch_out = torch.baddbmm(batch_add, batch_in, batch_mat)
    activation_mean = batch_out[0]
    activation_std = torch.sqrt(batch_out[1])

    epsilon = torch.normal(torch.zeros_like(activation_mean.shape), 1)
    return activation_mean + activation_std * epsilon

def init_conv(in_channels, out_channels, kernel_size):
    weight_mean = torch.empty((out_channels, in_channels, kernel_size, kernel_size)).normal_(0, 0.1)
    weight_std = torch.empty((out_channels, in_channels, kernel_size, kernel_size)).uniform_(-3, -3)
    return weight_mean, weight_std

def reparam_conv(input, weight_mean, weight_rho, stride, padding):
    weight_std = F.softplus(weight_rho)

    activation_mean = F.conv2d(input, weight_mean, None, stride=stride, padding=padding)
    activation_var = F.conv2d(input**2, weight_std**2, None, stride=stride, padding=padding)
    activation_std = torch.sqrt(activation_var)

    epsilon = torch.normal(torch.zeros_like(activation_mean), 1)
    return activation_mean + activation_std * epsilon

def reparam_conv_bias(input, weight_mean, weight_rho, bias_mean, bias_rho, stride, padding):
    weight_std = F.softplus(weight_rho)
    bias_std = F.softplus(bias_rho)

    activation_mean = F.conv2d(input, weight_mean, bias_mean, stride=stride, padding=padding)
    activation_var = F.conv2d(input**2, weight_std**2, bias_std**2, stride=stride, padding=padding)
    activation_std = torch.sqrt(activation_var)

    epsilon = torch.normal(torch.zeros_like(activation_mean), 1)
    return activation_mean + activation_std * epsilon



def gauss_kl(sigma0, mu1, sigma1):
    kl = 0.5 * (2 * torch.log(sigma0 / sigma1) - 1 + (sigma1 / sigma0).pow(2) + (mu1 / sigma0).pow(2))
    return kl.sum()