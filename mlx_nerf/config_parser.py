import argparse

def config_parser():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--basedir", type=str, default="./logs/", help="where to store checkpoints and logs")
    parser.add_argument("--datadir", type=str, default="./data/llff/fern", help="input data directory")

    # NOTE: tranining options
    # ------------------------------------------
    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, help="channels per layer")
    parser.add_argument("--netdepth_fine", type=int, default=8, help="layers in fine network")
    parser.add_argument("--netwidth_fine", type=int, default=256, help="channels per layer in fine network")
    parser.add_argument("--N_rand", type=int, default=32*32*4, help="batch size (number of random rays per gradient step)")
    parser.add_argument("--lrate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--lrate_decay", type=int, default=250, help="exponential learning rate decay (in 1000 steps)")
    
    ## NOTE: training options - batch size
    parser.add_argument("--chunk", type=int, default=1024*32, help="number of rays processed in parallel, decrease it if running out of memory")
    parser.add_argument("--netchunk", type=int, default=1024*64, help="number of points sent through network in parallel, decrease it if running out of memory")
    parser.add_argument("--no_batching", action="store_true", help="only take random rays from 1 image at a time")
    parser.add_argument("--no_reload", action="store_true", help="do not reload weights from saved checkpoint")
    parser.add_argument("--ft_path", type=str, default=None, help="specific weights npy file to reload for coarse network")

    ## NOTE: training options - precrop?
    parser.add_argument("--precrop_iters", type=int, default=0, help="number of steps to train on central crops")
    parser.add_argument("--precrop_frac", type=float, default=0.5, help="fraction of image taken for central crops")
    # ------------------------------------------


    # NOTE: rendering options
    # ------------------------------------------
    parser.add_argument("--n_depth_samples", type=int, default=64, help="number of coarse samples per ray")
    parser.add_argument("--N_importance", type=int, default=0, help="number of additional fine samples per ray")
    parser.add_argument("--perturb", type=float, default=1., help="set to 0. for no jitter, 1. for jitter")
    parser.add_argument("--use_viewdirs", action="store_true", help="use full 5D input instead of 3D")
    parser.add_argument("--i_embed", type=int, default=0, help="set 0 for default positional encoding, -1 for none")
    parser.add_argument("--multires", type=int, default=10, help="log2 of max freq for positional encoding (3D location)")
    parser.add_argument("--multires_views", type=int, default=4, help="log2 of max freq for positional encoding (2D direction)")
    parser.add_argument("--raw_noise_std", type=float, default=0., help="std dev of noise added to regularize sigma_a output. 1e0 recommended")

    ## NOTE: rendering options - visualization related?
    parser.add_argument("--render_only", action="store_true", help="do not optimize, reload weights and render out render_poses path")
    parser.add_argument("--render_test", action="store_true", help="render the test set instead of render_poses path")
    parser.add_argument("--render_factor", type=int, default=0, help="downsampling factor to speed up rendering, set 4 or 8 for fast preview")
    # ------------------------------------------

    # NOTE: dataset options
    # ------------------------------------------
    parser.add_argument("--dataset_type", type=str, default="llff", help="options: llff / blender / deepvoxels")
    parser.add_argument("--testskip", type=int, default=8, help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels")

    ## NOTE: dataset options - deepvoxels flags
    parser.add_argument("--shape", type=str, default="greek", help="options: armchair / cube / greek / vase")

    ## NOTE: dataset options - blender flags
    parser.add_argument("--white_bkgd", action="store_true", help="set to render synthetic data on a white background (always true for deepvoxels)")
    parser.add_argument("--half_res", action="store_true", help="load blender synthetic data at 400*400, instead of 800*800")

    ## NOTE: dataset options - LLFF flags
    parser.add_argument("--factor", type=int, default=8, help="downsample factor for LLFF images")
    parser.add_argument("--no_ndc", action="store_true", help="do not use normalized device coordinates (set for non-forward facing scenes)")
    parser.add_argument("--lindisp", action="store_true", help="sampling linearly in disparity rather than depth")
    parser.add_argument("--spherify", action="store_true", help="set for spherical 360deg scenes")
    parser.add_argument("--llffhold", type=int, default=8, help="will take every 1/N images as LLFF test set, paper uses 8")
    # ------------------------------------------
    
    # NOTE: Logging / Saving options
    # ------------------------------------------
    parser.add_argument("--i_print", type=int, default=100, help="frequency of console printout and metric logging")
    parser.add_argument("--i_img", type=int, default=500, help="frequency of tensorboard image logging")
    parser.add_argument("--i_weights", type=int, default=10000, help="frequency of weight checkpoint saving")
    parser.add_argument("--i_testset", type=int, default=50000, help="frequency of testset saving")
    parser.add_argument("--i_video", type=int, default=50000, help="frequency of render_poses video saving")
    # ------------------------------------------

    return parser

def load_config(filename_config: str = "configs/lego.txt"):

    with open(filename_config, "r") as fp:
        lines = fp.readlines()
        lines = [
            line.strip()
            for line in lines
        ]

        lines = [
            line.split(" = ")
            for line in lines
            if len(line) > 0
        ]

    configs = {}
    for line in lines:
        configs[line[0]] = line[1]

    return configs

# NOTE: per NeRF?
def update_NeRF_args(args: argparse.Namespace, configs: dict):

    args.expname = configs['expname']
    args.basedir = configs['basedir']
    args.datadir = configs['datadir']
    args.dataset_type = configs['dataset_type']
    args.no_batching = configs['no_batching']
    args.use_viewdirs = configs['use_viewdirs']
    args.white_bkgd = configs['white_bkgd']
    args.lrate_decay = int(configs['lrate_decay'])
    args.n_depth_samples = int(configs['N_samples'])
    args.N_importance = int(configs['N_importance'])
    args.N_rand = int(configs['N_rand'])
    args.precrop_iters = int(configs['precrop_iters'])
    args.precrop_frac = float(configs['precrop_frac'])
    args.half_res = configs['half_res']
    args.no_reload = True

    return args