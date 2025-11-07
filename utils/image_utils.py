#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'depth':
        net_image = render_pkg['depth']
    elif output == 'alpha':
        net_image = render_pkg["rend_alpha"]
    elif output == 'normal':
        net_image = render_pkg["normal_map"]
        net_image = (net_image+1)/2
    elif output == 'base color':
        net_image = render_pkg["base_color_map"]
    elif output == 'refl. strength':
        net_image = render_pkg["refl_strength_map"].repeat(3,1,1)
    elif output == 'refl. color':
        net_image = render_pkg["refl_color_map"]
    else:
        net_image = render_pkg["render"]

    return net_image