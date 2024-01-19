import mlx.nn as nn

class NeRF(nn.Module):
    def __init__(
        self, 
        n_layers=8, 
        width_layers=256, 
        channel_input=3,
        channel_input_views=3, 
        channel_output=4,
        list_skip_connection_layers=[4], 
        is_use_view_directions=False, # FIXME: deprecate; default option after all
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
        self.list_linears_dir = [nn.Linear(width_layers+channel_input_views, width_layers//2)]
        # fmt: on

        if True is is_use_view_directions:
            self.feature_linear = nn.Linear(width_layers, width_layers)
            self.alpha_linear = nn.Linear(width_layers, 1)
            self.rgb_linear = nn.Linear(width_layers//2, 3)
        else: # FIXME: deprecate
            self.output_layer = nn.Linear(width_layers, channel_output)

        return