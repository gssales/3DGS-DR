
from argparse import ArgumentParser
import sys
import traceback

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import PILtoTorch, safe_state
from utils.image_utils import render_net_image

def stage(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, iteration):
  
  gaussians = GaussianModel(dataset.sh_degree)
  scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
  gaussians.training_setup(opt)
  
  bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
  background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

  with torch.no_grad():
    viewpoint_cam = scene.getTrainCameras()[0]
    
    ## load mask
    img = Image.open('mask_00001.jpg')
    orig_w, orig_h = img.size
    resolution = round(orig_w / 6), round(orig_h / 6)
    mask = PILtoTorch(img, resolution).cuda()

    # render with mask
    render_pkg = render(viewpoint_cam, gaussians, pipe, background, initial_stage=False, img_mask=mask)
    image = render_pkg["render"]
    is_rendered = render_pkg["is_rendered"]

    image = image*mask

    print(torch.sum(is_rendered))
    torchvision.utils.save_image(image, 'image.png')
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    print("\nTesting rasterizer with img mask")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    stage(lp.extract(args), op.extract(args), pp.extract(args), args.iteration)

    # All done
    print("\nStaging complete.")