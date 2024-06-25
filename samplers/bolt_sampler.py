from .base_sampler import BaseSampler
from torch.optim import SGD
from transformers import AdamW
import torch.distributions as dists
import torch 


class BoltSampler(BaseSampler): 
    def __init__(self, 
                 batch_size, 
                 optimizer, 
                 optimizer_conf, 
                 noise_conf, 
                 device, **kwargs):
        super(BoltSampler, self).__init__()
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer_kw = optimizer_conf
        self.noise_kw = noise_conf
        self.device = device

    def initialize_batch(self, 
                         model, 
                         seq_length,
                         inputs, 
                         sentiment=None, 
                         **kwargs): 
        # initializing the biases for the model 
        model.set_biases(seq_len=seq_length, 
                         attribute=sentiment,
                         device=self.device,
                         **kwargs)
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
                                     **self.optimizer_kw)
        else: 
            self.cur_optimizer = AdamW(optimizer_grouped_parameters, 
                                       **self.optimizer_kw)
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
        loss.backward()
        self.cur_optimizer.step()
        noise = [torch.normal(
            size=model.biases[0].shape,
            requires_grad=False,
             **self.noise_kw
            ).to(self.device)
            for _ in range(len(model.biases))
        ]
        for i in range(len(model.biases)):
            model.biases[i].data = model.biases[i].data + noise[i]
        return x, loss, output_ids, otheroutputs


        
