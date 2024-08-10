from .base_sampler import BaseSampler
from torch.optim import SGD, RMSprop, Adagrad
from transformers import AdamW
import torch.distributions as dists
import torch 
import numpy as np


class BoltSampler(BaseSampler): 
    def __init__(self, 
                 batch_size, 
                 optimizer, 
                 optimizer_conf, 
                 noise_conf, 
                 device, is_kw=False, use_kw_embed_init=False, **kwargs):
        super(BoltSampler, self).__init__()
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer_kw = optimizer_conf
        self.noise_kw = noise_conf
        self.device = device
        self.grad_norms = []
        self.cur_grad_norms = []
        self.bias_max_val = []
        self.bias_norms = []
        self.losses_to_store = []
        self.cur_losses_to_store = None
        self.cur_bias_norms = None 
        self.is_kw = is_kw
        self.use_kw_embed_init = use_kw_embed_init
        self.angle_changes = []
        self.cur_angle_changes = None

    def initialize_batch(self, 
                         model, 
                         seq_length,
                         inputs, 
                         sentiment=None,
                         keyword_tokens=None,
                         **kwargs): 
        # initializing the biases for the model
        if self.cur_losses_to_store is not None:
            self.losses_to_store.append(self.cur_losses_to_store)
        self.cur_losses_to_store = []
        if self.use_kw_embed_init:
            print(keyword_tokens[0, 0])
            kw_embed= model.get_input_embeddings()(keyword_tokens[0, :].sum(dim=0))
        else: 
            kw_embed= None
        if self.cur_bias_norms is not None:
            self.bias_norms.append(self.cur_bias_norms)
        self.cur_bias_norms = []
        if self.cur_angle_changes is not None:
            self.angle_changes.append(self.cur_angle_changes)
        self.prev_bias_tensor = []
        self.cur_angle_changes = []
        model.set_biases(seq_len=seq_length, 
                         attribute=sentiment,
                         device=self.device,
                         kw_embed=kw_embed,
                         **kwargs)
        if self.cur_grad_norms is not None: 
            self.grad_norms.append(self.cur_grad_norms)
        self.cur_grad_norms = []
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "biases" in n or "trainable_weights" in n
                ],
                "weight_decay": self.optimizer_kw['weight_decay']
            }
        ]
        if self.optimizer == "sgd": 
            self.cur_optimizer = SGD(optimizer_grouped_parameters, 
                                    lr= self.optimizer_kw['lr'],)
        else:
            if self.optimizer == 'rmsprop':
                OptClass = RMSprop
            elif self.optimizer == 'adagrad':
                OptClass = Adagrad
            else: 
                OptClass = AdamW
            self.cur_optimizer = OptClass(optimizer_grouped_parameters, 
                                       **self.optimizer_kw)
        print(self.cur_optimizer)
        return inputs, optimizer_grouped_parameters
    
    def step(self, 
             x,
             model,
             inputs,
             **kwargs): 
        self.cur_optimizer.zero_grad()
        loss, output_ids, *otheroutputs = model.soft_forward(
                    **inputs, 
                    labels=inputs, 
                    use_full_prompt=False,
                    **kwargs
                )
        if self.is_kw: 
            self.cur_losses_to_store.append(otheroutputs[0])
        loss.backward()

        cur_biases_grad_norm = []
        # tracking grad norms
        cur_bias_tensor = [] 
        for i in range(len(model.biases)):
            grad = model.biases[i].grad
            if grad is not None: 
                grad_norm = grad.detach().data.norm(2).cpu().numpy()
                cur_biases_grad_norm.append(grad_norm)
                cur_bias_tensor.append(model.biases[i].detach().clone())
        cur_bias_tensor = torch.stack(cur_bias_tensor)
        if self.prev_bias_tensor != []: 
            angle_change = torch.einsum('bse, bse -> bs', [cur_bias_tensor, self.prev_bias_tensor]) / (cur_bias_tensor.norm(2, dim=-1) * self.prev_bias_tensor.norm(2, dim=-1))
            self.cur_angle_changes.append(angle_change)
        self.prev_bias_tensor = cur_bias_tensor
        self.cur_grad_norms.append(np.stack(cur_biases_grad_norm))
        self.cur_optimizer.step()
        noise = [torch.normal(
            size=model.biases[0].shape,
            requires_grad=False,
             **self.noise_kw
            ).to(self.device)
            for _ in range(len(model.biases))
        ]
        cur_bias_norms = []
        cur_max_vals = []
        for i in range(len(model.biases)):
            cur_bias_norms.append(model.biases[i].norm(2).detach().cpu().numpy())
            cur_max_vals.append(torch.max(model.biases[i].detach(), dim=-1).values.cpu())
            model.biases[i].data = model.biases[i].data + noise[i]
        self.bias_max_val.append(cur_max_vals)
        self.cur_bias_norms.append(np.stack(cur_bias_norms))

        return x, loss, output_ids, otheroutputs
    
    def get_sampling_metrics(self): 
        if self.cur_grad_norms is not None: 
            self.grad_norms.append(self.cur_grad_norms)
        if self.cur_losses_to_store is not None:
            self.losses_to_store.append(self.cur_losses_to_store)
        if self.cur_bias_norms is not None:
            self.bias_norms.append(self.cur_bias_norms)
        if self.cur_angle_changes is not None:
            self.angle_changes.append(self.cur_angle_changes)
        return_dct = {
                'bias_max_vals': self.bias_max_val,
                'grad_norms': self.grad_norms, 
                'bias_norms': self.bias_norms,
                'angle_changes': self.angle_changes,
                }
        if self.is_kw: 
            return_dct['losses'] = self.losses_to_store 
        return return_dct

        
