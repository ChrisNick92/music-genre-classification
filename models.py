import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_features=26, output_features=4):
        super(MLP, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.layer_dimensions = [input_features] + [128, 32] + [output_features]

        self._build_layers()

    def _build_layers(self):
        self.body = nn.ModuleList()
        for i, d in enumerate(self.layer_dimensions[:-1]):
            self.body.append(nn.Linear(in_features=d, out_features=self.layer_dimensions[i + 1]))

    def forward(self, x):
        for op in self.body:
            x = op(x)
            
        return x