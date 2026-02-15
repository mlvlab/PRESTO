from transformers import LlamaForSequenceClassification, LlamaModel, LlamaPreTrainedModel, LlamaForCausalLM
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from backpack import backpack, extend
from backpack.extensions import BatchGrad
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.multivariate_normal import MultivariateNormal
import math
import time
tkwargs = {
    "device": torch.device("cuda:0"),
    "dtype": torch.float32,
}

def linear_rampup(t, T_ramp, lambda_max):
    if t >= T_ramp:
        return lambda_max
    return lambda_max * (t / T_ramp)

class Network(nn.Module):
    def __init__(self, input_dim, hidden_size=100, depth=1, init_params=None):
        super(Network, self).__init__()

        self.activate = nn.ReLU()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, hidden_size))
        for i in range(depth-1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))
    
    def forward(self, x):
        y = x
        for i in range(len(self.layer_list)-1):
            y = self.activate(self.layer_list[i](y))
        y = self.layer_list[-1](y)
        return y   
    
class NeuralTSDiag:
    def __init__(self, input_dim, lamdba=1, nu=1, style='ucb', init_x=None, init_y=None, diagonalize=True, g_lambda=0.1):
        self.diagonalize = diagonalize
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.func = extend(Network(input_dim).to(**tkwargs))
        self.init_state_dict = deepcopy(self.func.state_dict())
        
        if init_x is not None:
            self.context_list = init_x.to(**tkwargs)
        else:
            self.context_list = None
        if init_y is not None:
            self.reward = init_y.to(**tkwargs)
        else:
            self.reward = None
        self.len = 0
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)

        if self.diagonalize:
            self.U = lamdba * torch.ones((self.total_param,))
        else:
            self.U = lamdba * torch.diag(torch.ones((self.total_param,)))
        
        self.nu = nu
        self.style = style
        self.loss_func = nn.MSELoss()
        self.mean = None
        self.std = None
        self.g_lambda = g_lambda
        self.beta=1.0

    def select(self, context, batch_size=300):     
        if self.mean is not None:
            context_ = (context - self.mean) / self.std   
        else:
            context_ = context

        context_size = context_.shape[0]        
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        g_list = []
        mu = []

        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context_[(i*batch_size):]
            else:
                context_batch = context_[(i*batch_size):((i+1)*batch_size)]
            mu_ = self.func(context_batch)
            sum_mu = torch.sum(mu_)
            with backpack(BatchGrad()):
                sum_mu.backward()                
            g_list_ = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)
            g_list.append(g_list_.cpu())
            mu.append(mu_.cpu())

        g_list = torch.vstack(g_list)
        mu = torch.vstack(mu)
        sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * g_list * g_list / self.U, dim=1))
        sample_r = mu.view(-1) + sigma.view(-1)
        arm = torch.argmax(sample_r)
        self.U += g_list[arm] * g_list[arm]

        return arm

    def update_U(self, context, batch_size=300):     
        if self.mean is not None:
            context_ = (context - self.mean) / self.std   
        else:
            context_ = context
            
        context_size = context_.shape[0]        
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        g_list = []
        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context_[(i*batch_size):]
            else:
                context_batch = context_[(i*batch_size):((i+1)*batch_size)]
            mu_ = self.func(context_batch)
            sum_mu = torch.sum(mu_)
            with backpack(BatchGrad()):
                sum_mu.backward()                
            g_list_ = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)
            g_list.append(g_list_.cpu())

        g_list = torch.vstack(g_list)
        self.U += (g_list * g_list).sum(dim=0)

    def train(self, context, reward, instruct_to_embedding, local_training_iter=30):
        if self.init_state_dict is not None:
            self.func.load_state_dict(deepcopy(self.init_state_dict))
        if context is not None:
            if self.context_list is None:
                self.context_list = torch.from_numpy(context.reshape(1, -1)).to(**tkwargs)
                self.reward = torch.tensor([reward]).to(**tkwargs)
            else:
                self.context_list = torch.cat((self.context_list, context.reshape(-1, self.context_list.shape[-1]).to(**tkwargs)))
                self.reward = torch.cat((self.reward, reward.to(**tkwargs)))

        optimizer = torch.optim.Adam(self.func.parameters(), lr=1e-4)

        self.std = self.context_list.std(dim=0) + 1e-30
        self.mean = self.context_list.mean(dim=0)
        standardized_context = (self.context_list - self.mean) / self.std 
        standardized_reward = self.reward.reshape(-1)
        
        all_embeddings = torch.cat(list(instruct_to_embedding.values()), dim=0)
        all_embeddings = (all_embeddings - self.mean) / self.std

        split_sizes = [v.size(0) for v in instruct_to_embedding.values()]
        group_ids = torch.cat([
            torch.full((sz,), idx, dtype=torch.long, device=all_embeddings.device)
            for idx, sz in enumerate(split_sizes)
        ])
        K = len(split_sizes)

        all_inputs = torch.cat((standardized_context, all_embeddings))
        supervised_idx = len(standardized_context)

        group_count = torch.tensor(split_sizes, device=all_embeddings.device)
        for epoch in range(local_training_iter):
            self.func.zero_grad()
            optimizer.zero_grad()
            pred_all = self.func(all_inputs).view(-1)
            pred = pred_all[:supervised_idx]
            all_preds = pred_all[supervised_idx:]
            mse_loss = self.loss_func(pred, standardized_reward)

            group_sum     = torch.zeros(K, device=all_preds.device, dtype=all_preds.dtype).scatter_add_(0, group_ids, all_preds)
            group_sum_sq  = torch.zeros(K, device=all_preds.device, dtype=all_preds.dtype).scatter_add_(0, group_ids, all_preds**2)

            num_pairs = group_count * (group_count - 1) / 2
            pairwise_sum = group_count * group_sum_sq - group_sum**2  # sum_{i<j}(g_i-g_j)^2 = n*sumg_i^2 - (sum g_i)^2 > n is the size of each group
            group_losses = pairwise_sum / num_pairs
            g_lambda = linear_rampup(epoch, local_training_iter * 0.5, self.g_lambda)
            group_loss = g_lambda * group_losses.mean() 

            loss = mse_loss + group_loss
            loss.backward()
            optimizer.step()
        
        return self.func.state_dict()