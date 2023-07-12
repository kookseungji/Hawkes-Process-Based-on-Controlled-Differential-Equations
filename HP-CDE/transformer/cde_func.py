import torch

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.elu = torch.nn.ELU(inplace=False)
        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linear_in2 = torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, t, z):
        z = self.linear_in(z)
        z = self.elu(z)
        z= self.linear_in2(z)
        for linear in self.linears:
            z = linear(z)
            z = self.elu(z)
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z


class ODEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, beta):
        
        super(ODEFunc, self).__init__()
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.beta = beta

        self.elu = torch.nn.ELU()
        
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    def softplus(self, x, beta):
         
        temp = beta * x
        temp[temp > 20] = 20
        return 1.0 / beta * torch.log(1 + torch.exp(temp))
        
    def forward(self, t, z): 

        z = self.linear(z) 
        z = self.softplus(z, self.beta)
        return z
