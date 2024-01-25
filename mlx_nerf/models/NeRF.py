import mlx.core as mx
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
            self.feature_linear = nn.Linear(width_layers, width_layers) # NOTE: last layer
            self.alpha_linear = nn.Linear(width_layers, 1)
            self.rgb_linear = nn.Linear(width_layers//2, 3)
        else: # FIXME: deprecate
            self.output_linear = nn.Linear(width_layers, channel_output)

        return
    
    def forward(
        self, 
        x # NOTE: encoded
    ):
        input_pos, input_dir = mx.split(
            x, 
            [self.channel_input_pos, self.channel_input_dir], 
            dim=-1
        )

        # NOTE: forwarding positions
        h = input_pos
        for idx, layer_pos in enumerate(self.list_linears_pos):

            h = layer_pos(h)
            h = nn.relu(h)

            if idx in self.list_skip_connection_layers:
                h = mx.concatenate([input_pos, h], dim=-1) # NOTE: skip connection

        # NOTE: forwarding directions
        # NOTE: refactor to be more readable
        if self.is_use_view_directions:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = mx.concatenate([feature, input_dir], dim=-1)

            for idx, layer_dir in enumerate(self.list_linears_dir):
                h = layer_dir(h)
                h = nn.relu(h)

            rgb = self.rgb_linear(h)
            outputs = mx.concatenate([rgb, alpha], dim=-1)
        else:
            outputs = self.output_linear(h)

        return outputs