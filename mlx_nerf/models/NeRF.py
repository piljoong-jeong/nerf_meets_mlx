from copy import deepcopy

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_nerf.models import embedding

def inference_wrapper_batch(model, chunk):

    if chunk is None:
        return model
    
    def __batched_model_inference(inputs_embedded):
        return mx.concatenate(
            [
                model(inputs_embedded[i:i+chunk])
                for i in range(0, inputs_embedded.shape[0], chunk)
            ], axis=0
        )
    return __batched_model_inference


def run_model(
    pos, embed_pos, 
    dir, embed_dir, 
    model, 
    netchunk = 64*1024
):
    # NOTE: embed `pos` & `dir`, and concatenate
    inputs_embedded = embedding.embed(pos, embed_pos, dir, embed_dir)

    # NOTE: batched inference & concatenate per batch
    outputs_flat = inference_wrapper_batch(model, netchunk)(inputs_embedded)
    
    # NOTE: reshape `outputs_flat` to have shape of `inputs_embedded`
    # TODO: double-check shape
    return (
        outputs := mx.reshape(
            outputs_flat, 
            list(inputs_embedded.shape[-1]) + [outputs_flat.shape[-1]]
        )
    )


def create_NeRF(args):
    """
    Returns coarse (& fine) NeRF models
    """

    # TODO: refactor `args`
    octave_pos = args.multires
    octave_dir = args.multires_views
    is_use_dir = args.use_viewdirs
    n_samples = args.N_samples
    n_importance_samples = args.N_importance
    learning_rate = args.lrate
    perturb = args.perturb
    raw_noise_std = args.raw_noise_std

    output_ch = 5 if n_importance_samples else 4 # TODO: what was the reason?
    # skips = args.skips
    skips = [4]

    # NOTE: embed each samples
    embedder_pos, channel_emb_pos = embedding.get_embedder(octave_pos) if True else (None, None)
    embedder_dir, channel_emb_dir = embedding.get_embedder(octave_dir) if is_use_dir else (None, None)

    # NOTE: define query function that internally batches
    network_query_fn = lambda inputs, viewdirs, model: run_model(
        inputs, embedder_pos, 
        viewdirs, embedder_dir, 
        model, 
        netchunk=args.netchunk
    )

    # NOTE: coarse NeRF
    n_layers = args.netdepth
    width_layers = args.netwidth
    # fmt: off
    model = NeRF(
        n_layers=n_layers, 
        width_layers=width_layers, 
        channel_input=channel_emb_pos, 
        channel_output=output_ch, 
        list_skip_connection_layers=skips, 
        channel_input_views=channel_emb_dir, 
        is_use_view_directions=is_use_dir
    )
    # fmt: on
    # TODO: deal with `grad_vars` in `loss_and_grad_fn`
    # grad_vars = list(model.parameters())


    # TODO: fine NeRF
    n_layers_fine = args.netdepth_fine
    width_layers_fine = args.netwidth_fine
    # fmt: off
    model_fine = NeRF(
        n_layers=n_layers_fine, 
        width_layers=width_layers_fine, 
        channel_input=channel_emb_pos, 
        channel_output=output_ch, 
        list_skip_connection_layers=skips, 
        channel_input_views=channel_emb_dir, 
        is_use_view_directions=is_use_dir
    ) if n_importance_samples > 0 else None
    # fmt: on
    if not None is model_fine:
        # TODO: deal with `grad_vars` in `loss_and_grad_fn`
        # grad_vars += list(model_fine.parameters())
        pass

    # FIXME: `mx.optimizers` does not accept `params`!
    optimizer = optim.Adam(learning_rate=learning_rate, betas=(0.9, 0.999))

    # TODO: load state here
    idx_iter = 0

    # TODO: load state here

    # NOTE: train arguments
    render_kwargs_train = {
        # NOTE: common
        "use_viewdirs": is_use_dir,
        "white_bkgd": args.white_bkgd, 
        "network_query_fn": network_query_fn, 
        
        # NOTE: coarse
        "network_fn": model, 
        "N_samples": n_samples, 

        # NOTE: fine
        "network_fine": model_fine, 
        "perturb": perturb, 
        "raw_noise_std": raw_noise_std, 
        "N_importance": n_importance_samples, 
    }
    if args.dataset_type != "llff" or args.no_ndc:
        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = args.lindisp

    # NOTE: test arguments
    render_kwargs_test = deepcopy(render_kwargs_train)
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0

    return render_kwargs_train, render_kwargs_test, idx_iter, optimizer

class NeRF(nn.Module):
    def __init__(
        self, 
        n_layers=8, 
        width_layers=256, 
        channel_input=3,
        channel_input_views=3, 
        channel_output=4,
        list_skip_connection_layers=[4], 
        is_use_view_directions=False, # NOTE: used when no views are given (say, image evaluation)
    ):
        super().__init__()

        self.D = n_layers
        self.W = width_layers
        self.channel_input_pos = channel_input
        self.channel_input_dir = channel_input_views
        self.list_skip_connection_layers = list_skip_connection_layers
        self.is_use_view_directions = is_use_view_directions

        # NOTE: layers
        # fmt: off
        self.list_linears_pos = [
            nn.Linear(channel_input, width_layers)
        ] + [
            nn.Linear(width_layers, width_layers) if i not in self.list_skip_connection_layers else
            nn.Linear(width_layers+channel_input, width_layers)
            for i in range(n_layers-1)
        ]
        # fmt: on

        if True is is_use_view_directions:
            self.list_linears_dir = [nn.Linear(width_layers+channel_input_views, width_layers//2)]
            self.feature_linear = nn.Linear(width_layers, width_layers) # NOTE: last layer
            self.alpha_linear = nn.Linear(width_layers, 1)
            self.rgb_linear = nn.Linear(width_layers//2, 3)
        else: # NOTE: e.g., image pixel learning
            self.output_linear = nn.Linear(width_layers, channel_output)

        return
    
    def forward(
        self, 
        x # NOTE: encoded
    ):

        if self.is_use_view_directions:
            input_pos, input_dir = mx.split(
                x, 
                indices_or_sections=[self.channel_input_pos, self.channel_input_dir], 
                axis=-1
            )
        else:
            input_pos = x

        # NOTE: forwarding positions
        h = input_pos
        for idx, layer_pos in enumerate(self.list_linears_pos):

            h = layer_pos(h)
            h = nn.relu(h)

            if idx in self.list_skip_connection_layers:
                h = mx.concatenate([input_pos, h], axis=-1) # NOTE: skip connection

        # NOTE: forwarding directions
        # NOTE: refactor to be more readable
        if self.is_use_view_directions:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = mx.concatenate([feature, input_dir], axis=-1)

            for idx, layer_dir in enumerate(self.list_linears_dir):
                h = layer_dir(h)
                h = nn.relu(h)

            rgb = self.rgb_linear(h)
            outputs = mx.concatenate([rgb, alpha], axis=-1)
        else:
            outputs = self.output_linear(h)

        return outputs