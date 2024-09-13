import argparse
from pkgutil import get_data
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import HPCDEv1
from tqdm import tqdm
import os

cur_path = os.path.dirname(__file__)

def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1') 
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    validloader = get_dataloader(dev_data, opt.batch_size, shuffle=False)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, validloader, testloader, num_types

def padding(emb):
    new_emb = emb.clone()
    for i in range(emb.shape[0]):
        sel_emb = emb[i]
        not_zero_sel_emb = sel_emb[sel_emb!=0]
        new_emb[i][sel_emb==0] = not_zero_sel_emb[-1]
    return new_emb

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0   
    total_time_se = 0   
    total_event_rate = 0   
    total_num_event = 0   
    total_num_pred = 0   
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        real_non_pad_mask = get_non_pad_mask(event_type)
        padded_event_time = padding(event_time)
        padded_event_type = padding(event_type)
        
        """ forward """
        optimizer.zero_grad()
        enc_out, ode_ouptut, prediction = model(padded_event_type, padded_event_time, real_non_pad_mask)

        """ backward """
         
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, ode_ouptut, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)    
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func, f1=False)

        se, total_num = Utils.time_loss(prediction[1], event_time, real_non_pad_mask)

         
        scale_time_loss = opt.scale
        scale_ll_loss = opt.llscale
        loss = (event_loss / scale_ll_loss) + pred_loss + (se /scale_time_loss)
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        total_num_pred += total_num.item()

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0   
    total_time_se = 0   
    total_event_rate = 0   
    total_num_event = 0   
    total_num_pred = 0   
    total_pred_type, total_correct_type = None, None
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
            real_non_pad_mask = get_non_pad_mask(event_type)
            padded_event_time = padding(event_time)
            padded_event_type = padding(event_type)

            """ forward """
            enc_out, ode_output, prediction = model(padded_event_type, padded_event_time, real_non_pad_mask)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, ode_output, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num, pred_type, correct_type = Utils.type_loss(prediction[0], event_type, pred_loss_func, f1=True)
            se, total_num = Utils.time_loss(prediction[1], event_time, real_non_pad_mask)
            """ for F1"""
            if total_pred_type is None:
                total_pred_type, total_correct_type = pred_type, correct_type
            else:
                if total_pred_type.shape[1] != pred_type.shape[1]:
                    if total_pred_type.shape[1] > pred_type.shape[1]:
                        size_match = torch.zeros(pred_type.shape[0],total_pred_type.shape[1]-pred_type.shape[1]).cuda()
                        pred_type = torch.cat([pred_type, size_match], dim=1)
                        correct_type = torch.cat([correct_type, size_match-1], dim=1)
                    else:
                        size_match = torch.zeros(total_pred_type.shape[0],pred_type.shape[1]-total_pred_type.shape[1]).cuda()
                        total_pred_type = torch.cat([total_pred_type, size_match], dim=1)
                        total_correct_type = torch.cat([total_correct_type, size_match-1], dim=1)

                total_pred_type = torch.cat([total_pred_type, pred_type], dim = 0)
                total_correct_type = torch.cat([total_correct_type, correct_type], dim = 0)
            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += total_num.item()

    predTypes = total_pred_type[total_correct_type!=-1]
    trueTypes = total_correct_type[total_correct_type!=-1]
    predTypes, trueTypes = predTypes.detach().cpu(), trueTypes.detach().cpu()
    f1score = f1_score(trueTypes,predTypes, average = 'macro')
    rmse = np.sqrt(total_time_se / total_num_pred)

    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, f1score


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """
    
    worse_step = 0
    early_stopped = False
    best_train_event, best_train_type, best_train_time = -1000000, 0, 10000000

    best_valid_event_ev, best_valid_type_ev, best_valid_time_ev, epoch_ev, best_f1_ev = 0, 0, 0, 0, 0
    best_valid_event_ty, best_valid_type_ty, best_valid_time_ty, epoch_ty, best_f1_ty = 0, 0, 0, 0, 0
    best_valid_event_ti, best_valid_type_ti, best_valid_time_ti, epoch_ti, best_f1_ti = 0, 0, 0, 0, 0

    valid_event_losses = []   
    valid_pred_losses = []   
    valid_rmse = []   
    valid_f1s = []
    print(f"[SCALE] : {opt.scale} | [LLSCALE] : {opt.llscale}\n")
    for epoch_i in range(opt.epoch):
        worse_step += 1
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

                
        torch.cuda.reset_max_memory_allocated(opt.device)
        baseline_memory = torch.cuda.memory_allocated(opt.device)

        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_type, valid_time, valid_f1 = eval_epoch(model, validation_data, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, F1 : {f1: 8.5f} '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, f1=valid_f1, elapse=(time.time() - start) / 60))

        memory_usage = torch.cuda.max_memory_allocated(opt.device) - baseline_memory
        print(f"memory_usage:{memory_usage}")

     
        if best_train_event < train_event:
            best_train_event = train_event
            best_valid_event_ev = valid_event
            best_valid_type_ev = valid_type
            best_valid_time_ev = valid_time
            epoch_ev = epoch
            best_f1_ev = valid_f1

        if best_train_type < train_type:
            best_train_type = train_type
            best_valid_event_ty = valid_event
            best_valid_type_ty = valid_type
            best_valid_time_ty = valid_time
            epoch_ty = epoch
            best_f1_ty = valid_f1
            worse_step = 0

        if best_train_time > train_time:
            best_train_time = train_time
            best_valid_event_ti = valid_event
            best_valid_type_ti = valid_type
            best_valid_time_ti = valid_time
            best_f1_ti = valid_f1
            epoch_ti = epoch

     

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        valid_f1s += [valid_f1]
        print('\n  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}, Maximum F1: {f1: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse), f1=max(valid_f1s)))
              
        print('  - [Info [BEST TRAIN LL ({sc: 8.5f})] at epoch {ep}] ll: {event: 8.5f}, '
            'accuracy: {pred: 8.5f}, RMSE: {rmse: 8.5f}, F1: {f1: 8.5f}'
            .format(sc=best_train_event, ep= epoch_ev, event=best_valid_event_ev, pred=best_valid_type_ev, rmse=best_valid_time_ev, f1 = best_f1_ev))

        print('  - [Info [BEST TRAIN ACC ({sc: 8.5f})] at epoch {ep}] ll: {event: 8.5f}, '
            'accuracy: {pred: 8.5f}, RMSE: {rmse: 8.5f}, F1: {f1: 8.5f}'
            .format(sc=best_train_type, ep= epoch_ty, event=best_valid_event_ty, pred=best_valid_type_ty, rmse=best_valid_time_ty, f1 = best_f1_ty))

        print('  - [Info [BEST TRAIN RMSE ({sc: 8.5f})] at epoch {ep}] ll: {event: 8.5f}, '
            'accuracy: {pred: 8.5f}, RMSE: {rmse: 8.5f}, F1: {f1: 8.5f}\n'
            .format(sc=best_train_time, ep= epoch_ti, event=best_valid_event_ti, pred=best_valid_type_ti, rmse=best_valid_time_ti, f1 = best_f1_ti))

         
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}, {f1: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time, f1=valid_f1))

        if worse_step >= 5:
            print("Early Stopped!")
            early_stopped = True
            break
        
        print("Worse steps : ", worse_step)
        scheduler.step()

    with open(opt.log, 'a') as f:
        if early_stopped:
            f.write('Early Stopped!\n')

        f.write('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}, Maximum F1: {f1: 8.5f}\n'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse), f1=min(valid_f1s)))

        f.write('  - [Info [BEST TRAIN LL ({sc: 8.5f})] at epoch {ep}] ll: {event: 8.5f}, '
            'accuracy: {pred: 8.5f}, RMSE: {rmse: 8.5f}, F1: {f1: 8.5f}\n'
            .format(sc=best_train_event, ep= epoch_ev, event=best_valid_event_ev, pred=best_valid_type_ev, rmse=best_valid_time_ev, f1 = best_f1_ev))

        f.write('  - [Info [BEST TRAIN ACC ({sc: 8.5f})] at epoch {ep}] ll: {event: 8.5f}, '
            'accuracy: {pred: 8.5f}, RMSE: {rmse: 8.5f}, F1: {f1: 8.5f}\n'
            .format(sc=best_train_type, ep= epoch_ty, event=best_valid_event_ty, pred=best_valid_type_ty, rmse=best_valid_time_ty, f1 = best_f1_ty))

        f.write('  - [Info [BEST TRAIN RMSE ({sc: 8.5f})] at epoch {ep}] ll: {event: 8.5f}, '
            'accuracy: {pred: 8.5f}, RMSE: {rmse: 8.5f}, F1: {f1: 8.5f}'
            .format(sc=best_train_time, ep= epoch_ti, event=best_valid_event_ti, pred=best_valid_type_ti, rmse=best_valid_time_ti, f1 = best_f1_ti))

        f.close()


def main():
    """ Main function. """
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',default='data/data_mimic/')
    parser.add_argument('--model', type=str, default='hpcdev1')
    parser.add_argument('--log', type=str, default='log.txt')
    parser.add_argument('--seed', type=int, default=2022)

    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--smooth', type=float, default=0.1)

    parser.add_argument('--scale', type=float, default=100)
    parser.add_argument('--llscale', type=float, default=100)

    parser.add_argument('--d_model', type=int, default=70)
    parser.add_argument('--d_ncde', type=int, default=128)
    parser.add_argument('--hh_dim', type=int, default=120)
    parser.add_argument('--n_layers', type=int, default=5)

    opt = parser.parse_args()

    manual_seed = opt.seed

    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.random.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    opt.device = torch.device('cuda')

     
    with open(opt.log, 'w') as f:
        f.write(f'Epoch, Log-likelihood, Accuracy, RMSE, F1 (Scale : {opt.scale}, LLScale : {opt.llscale})\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, validloader, testloader, num_types = prepare_dataloader(opt)

    """ prepare model """
    if opt.model == 'hpcdev1':
        model = HPCDEv1(
            num_types=num_types,
            d_model=opt.d_model,
            d_ncde = opt.d_ncde,
            hidden_hidden_dim = opt.hh_dim,
            n_layers = opt.n_layers,
        )

    model.to(opt.device)

    print(model)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                        opt.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay = opt.decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt)

    

if __name__ == '__main__':
    main()
