import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    """
    Standard multi-layer perceptron class. Has compatability with methods needed
    by AOP implementations of algorithms (select_action).
    """

    def __init__(self, params, use_bn=False):
        super(MLP, self).__init__()
        self.params = params

        if self.params['activation'] == 'relu':
            self.activation = torch.relu
        elif self.params['activation'] == 'tanh':
            self.activation = torch.tanh
        else:
            print('WARNING: activation not recognized, using relu')
            self.activation = torch.relu

        layer_sizes = [self.params['input_size']] + \
                      self.params['hidden_sizes'] + \
                      [self.params['output_size']]

        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))

        self.layers = nn.ModuleList(self.layers)

        self.dropout = nn.Dropout(p=self.params['dropout'])

        if 'dtype' in self.params:
            self.dtype = self.params['dtype']
        else:
            self.dtype = torch.float32

        for i in range(len(self.layers)):
            layer = self.layers[i]
            nn.init.kaiming_normal_(layer.weight)
            nn.init.normal_(layer.bias, 0)

    def forward(self, x):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
            if layer != self.layers[-1]:
                x = self.activation(x)
                x = self.dropout(x)
        if 'final_activation' in self.params:
            x = self.params['final_activation'](x)
        return x

    def select_action(self, x):
        x = torch.tensor(x, dtype=self.dtype)
        y = torch.squeeze(self.forward(x), dim=-1)
        return y.detach().cpu().numpy()
