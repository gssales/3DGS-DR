import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, network_gui
import torch
import matplotlib.pyplot as plt

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'depth':
        net_image = render_pkg['depth']
    elif output == 'normal':
        net_image = render_pkg["normal_map"]
        net_image = (net_image+1)/2
    elif output == 'base color':
        net_image = render_pkg["base_color_map"]
    elif output == 'refl. strength':
        net_image = render_pkg["refl_strength_map"] #.repeat(3,1,1)
    elif output == 'refl. color':
        net_image = render_pkg["refl_color_map"]
    else:
        net_image = render_pkg["render"]

    if net_image.shape[0]==1:
        net_image = colormap(net_image)
    return net_image

def view(dataset, pipe, iteration):
    
    view_render_options = ["RGB", "Base Color", "Refl. Strength", "Normal", "Refl. Color"]

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=[])
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    while True:
        with torch.no_grad():
            if network_gui.conn == None:
                network_gui.try_connect(view_render_options)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, view_render_options, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "_METHOD": "3DGS-DR",
                        "#": gaussians.get_opacity.shape[0],
                        "it": iteration
                        # Add more metrics as needed
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                except Exception as e:
                    raise e
                    print('Viewer closed')
                    exit(0)

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--iteration', type=int, default=30000)
    args = get_combined_args(parser)
    print("View: " + args.model_path)
    print("View: ", args)
    network_gui.init(args.ip, args.port)
    
    view(lp.extract(args), pp.extract(args), args.iteration)

    print("\nViewing complete.")