import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import get_non_pad_mask

def softplus(x, beta):
   
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))

def compute_event(event, non_pad_mask):
    
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result

def compute_integral_unbiased(model, ode_output, time, non_pad_mask, type_mask):
     
    temp_hid = torch.sum(ode_output[:,-1:,:] * type_mask[:, -1:, :], dim=2, keepdim=True)
    non_event_ll = torch.sum(temp_hid, dim=2)  
    
    return non_event_ll  

def log_likelihood(model, data, ode_output, time, types):
     

    non_pad_mask = get_non_pad_mask(types).squeeze(2)
   
    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    all_hid = model.linear(data)
    all_lambda = softplus(all_hid, model.beta)  
    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

     
    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)     
     
    non_event_ll = compute_integral_unbiased(model, ode_output, time, non_pad_mask, type_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)  

    return event_ll, non_event_ll 


def type_loss(prediction, types, loss_func, f1=False):
    
    truth = types[:, 1:] - 1
    true_truth = truth.clone().detach()
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

     
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)

    if f1: return loss, correct_num, pred_type, true_truth
    else: return loss, correct_num


def time_loss(prediction, event_time, mask):
     

    mask = mask.squeeze(-1)
    prediction = prediction.squeeze(-1)

    mask = mask[:,1:]

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

     
    diff = prediction - true
    diff = diff * mask
    se = torch.sum(diff * diff)

    total_num = torch.sum(mask==1)
    return se, total_num

def log_likelihood_original(model, data, time, types, ode_out):
    """ Log-likelihood of sequence. """

    non_pad_mask = get_non_pad_mask(types).squeeze(2) 
    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device) 

    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)
    
    all_hid = model.linear(data) 
    all_lambda = softplus(all_hid, model.beta) 
    type_lambda = torch.sum(all_lambda * type_mask, dim=2) 
    
    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    ode_out = ode_out * type_mask
    ode_out = ode_out[:,-1,:]
    ode_out = torch.sum(ode_out, dim=1)  
     
    return event_ll, ode_out 

def type_loss_original(prediction, types, loss_func):
 
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth) 

    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num

def type_loss_edit_f1(prediction, types, loss_func):
    
    truth = types[:, 1:] - 1
    true_truth = truth.clone().detach()
    prediction = prediction[:, :-1, :]
    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)
    
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)
    loss = torch.sum(loss)
    return loss, correct_num, pred_type, true_truth
    
def time_loss_edit(prediction, event_time, mask):
   
    mask = mask.squeeze(-1)
    prediction = prediction.squeeze(-1)
    mask = mask[:,1:]
    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]
   
    diff = prediction - true
    diff = diff * mask
    se = torch.sum(diff * diff)
    total_num = torch.sum(mask==1)
    return se, total_num

class LabelSmoothingLoss(nn.Module):

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss


class EarlyStopping(): #ACCURACY
    
    def __init__(self, patience=5, min_delta=0):
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.best_loss = val_loss
            
            self.counter = 0
        elif self.best_loss - val_loss > self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True



class RunningAverageMeter(object):

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val