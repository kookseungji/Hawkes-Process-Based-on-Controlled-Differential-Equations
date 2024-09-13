import math
import torch
import torch.nn as nn
import torchcde as torchcde
from transformer.cde_func import CDEFunc, ODEFunc
import transformer.Constants as Constants


def get_non_pad_mask(seq):
     
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


class Embeddings(nn.Module):
     

    def __init__(self, num_types, d_model):
        super().__init__()

        self.d_model = d_model
         
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

         
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

    def temporal_enc(self, time, non_pad_mask):

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
 
        tem_enc = self.temporal_enc(event_time, non_pad_mask)  
        enc_output = self.event_emb(event_type)  

        return enc_output, tem_enc

class Predictor(nn.Module):

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out

class Predictor2(nn.Module):
     
    def __init__(self, dim, num_types):
        super().__init__()

        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, num_types, bias=False)
        self.elu = torch.nn.ELU(inplace=True)

        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)


    def forward(self, data, non_pad_mask):
        out = self.linear1(data)
        out = self.elu(out)
        out = self.linear2(out)
        out = out * non_pad_mask
        return out



class HPCDEv1(nn.Module):
    def __init__(
            self,
            num_types, d_model=256,
            d_ncde = 512, hidden_hidden_dim = 256, n_layers = 2):
        super().__init__()

        self.get_embedding = Embeddings(num_types=num_types, d_model=d_model)
        self.num_types = num_types
        self.beta = nn.Parameter(torch.tensor(1.0))

        input_dim = d_model + 1
        self.initial = nn.Linear(d_model + 1, d_ncde)
        self.readout = nn.Linear(d_ncde, d_model)

        self.linear = nn.Linear(d_model, num_types)
        self.type_predictor = Predictor2(d_model, num_types)
        self.time_predictor = Predictor(d_model, 1)

        self.d_ncde = d_ncde
        self.hidden_hidden_dim = hidden_hidden_dim
        self.dim = d_ncde

        self.func = CDEFunc(input_dim,self.dim,self.hidden_hidden_dim, n_layers)
        self.odeFunc = ODEFunc(input_dim, self.dim, num_types, beta = self.beta)


    def forward(self, event_type, event_time, real_non_pad_mask):

        non_pad_mask = get_non_pad_mask(event_type)  
        event_emb, temp_emb = self.get_embedding(event_type, event_time, non_pad_mask)
        concat_emb = event_emb + temp_emb

        event_time = event_time.unsqueeze(-1)
        concat_emb = torch.cat([event_time,concat_emb], dim=2)

        coeffs = torchcde.linear_interpolation_coeffs(x = concat_emb).to(event_type.device)
        X = torchcde.LinearInterpolation(coeffs)

        h0 = self.initial(concat_emb[:,0,:])
        a0 = torch.zeros(h0.shape[0],self.num_types).to(event_type.device)

        adjoint_params = tuple(self.func.parameters()) + tuple(self.odeFunc.parameters()) + (coeffs,)

        enc_output, ode_output = torchcde.cdeint_augment(X=X, func = self.func, z0 = h0, a0 = a0, ode_func = self.odeFunc, event_time = event_time,
                                         t=X.grid_points, adjoint_params = adjoint_params,
                                         method = 'rk4')
        enc_output = self.readout(enc_output)
        enc_output = enc_output*real_non_pad_mask
        time_prediction = self.time_predictor(enc_output, real_non_pad_mask)  
        type_prediction = self.type_predictor(enc_output, real_non_pad_mask)  

        return enc_output, ode_output, (type_prediction, time_prediction)
