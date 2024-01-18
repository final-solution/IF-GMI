import math
import os
from typing import Tuple
import pickle

import torch
import torchvision
import torchvision.transforms.functional as F

import sys


def adjust_gen_images(imgs: torch.tensor,
                      bounds: Tuple[torch.tensor, torch.tensor], size: int):
    """
    Change the value range of images generated by StyleGAN2. Outputs are usually roughly in the range [-1, 1]. 
    A linear transformation is then applied following the transformation used in the official implementation to save images.
    Images are resized to a given size using bilinear interpolation.
    """
    lower_bound, upper_bound = bounds
    lower_bound = lower_bound.float().to(imgs.device)
    upper_bound = upper_bound.float().to(imgs.device)
    imgs = torch.where(imgs > upper_bound, upper_bound, imgs)
    imgs = torch.where(imgs < lower_bound, lower_bound, imgs)
    imgs = F.center_crop(imgs, (700, 700))
    imgs = F.resize(imgs, size, antialias=True)
    return imgs


def save_images(imgs: torch.tensor, folder, filename, center_crop=800):
    """Save StyleGAN output images in file(s).

    Args:
        imgs (torch.tensor): generated images in [-1, 1] range
        folder (str): output folder
        filename (str): name of the files
    """
    imgs = imgs.detach()
    if center_crop:
        imgs = F.center_crop(imgs, (center_crop, center_crop))
    imgs = (imgs * 0.5 + 128 / 255).clamp(0, 1)
    for i, img in enumerate(imgs):
        path = os.path.join(folder, f'{filename}_{i}.png')
        torchvision.utils.save_image(img, path)

# 对图像进行变换
def create_image(imgs, crop_size=None, resize=None):
    if crop_size is not None:
        imgs = F.center_crop(imgs, (crop_size, crop_size))
    if resize is not None:
        imgs = F.resize(imgs, resize, antialias=True)
    return imgs

# def create_image(w,
#                  generator,
#                  crop_size=None,
#                  resize=None,
#                  batch_size=20,
#                  device='cuda'):
#     with torch.no_grad():
#         if w.shape[1] == 1:
#             w_expanded = torch.repeat_interleave(w,
#                                                  repeats=generator.num_ws,
#                                                  dim=1)
#         else:
#             w_expanded = w

#         w_expanded = w_expanded.to(device)
#         imgs = []
#         for i in range(math.ceil(w_expanded.shape[0] / batch_size)):
#             w_batch = w_expanded[i * batch_size:(i + 1) * batch_size]
#             imgs_generated = generator(w_batch,
#                                        noise_mode='const',
#                                        force_fp32=True)
#             imgs.append(imgs_generated.cpu())

#         imgs = torch.cat(imgs, dim=0)
#         if crop_size is not None:
#             imgs = F.center_crop(imgs, (crop_size, crop_size))
#         if resize is not None:
#             imgs = F.resize(imgs, resize, antialias=True)
#         return imgs


def load_generator(filepath):
    """Load pre-trained generator using the running average of the weights ('ema').

    Args:
        filepath (str): Path to .pkl file

    Returns:
        torch.nn.Module: G_ema from torch.load
    """
    sys.path.append('stylegan2_intermediate')
    from stylegan2_intermediate.training.networks import Generator
    print('使用的GAN路径为:', filepath)
    mapping = {'num_layers': 8,
               'embed_features': None,
               'layer_features': None,
               'activation': 'lrelu',
               'lr_multiplier': 0.01,
               'w_avg_beta': 0.995}
    synthesis = {'channel_base': 32768, 'channel_max': 512, 'num_fp16_res': 4, 'conv_clamp': 256,
                 'architecture': 'skip', 'resample_filter': [1, 3, 3, 1], 'use_noise': True, 'activation': 'lrelu'}
    G = Generator(z_dim=512, c_dim=0, w_dim=512,
                  img_resolution=1024, img_channels=3,
                  mapping_kwargs=mapping, synthesis_kwargs=synthesis)
    # with open(filepath, 'rb') as f:
    #     obj = f.read()
    # weights = {key: weight_dict for key,
    #            weight_dict in pickle.loads(obj, encoding='latin1').items()}
    # G.load_state_dict(weights, strict=False)
    state_dict = torch.load(filepath, map_location='cpu')['state_dict']
    G.load_state_dict(state_dict)
    G = G.cuda()
    return G


def load_discrimator(filepath):
    """Load pre-trained discriminator

    Args:
        filepath (str): Path to .pkl file

    Returns:
        torch.nn.Module: D from pickle
    """
    with open(filepath, 'rb') as f:
        sys.path.insert(0, 'stylegan2_intermediate')
        D = pickle.load(f)['D'].cuda()
    return D


def project_onto_l1_ball(x, eps):
    """
    See: https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)
