import torch 
from .dlp_embed import LangevinSampler
import random 
import itertools


class StochasticLangevin(LangevinSampler): 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # number of bias terms to be non-zero 
        self.num_bias_positions = kwargs.get('num_bias_positions', 1)

        # spacing between bias terms
        self.bias_spacing = kwargs.get('bias_spacing', 1)

        # start cutoff for bias terms
        self.end_cutoff = self.num_bias_positions * self.bias_spacing + 1        

    # there is probably a better way to code this, 
    # but for now I am going to brute force it
    def construct_bias_mask(self, shape):
        (batch_size, seq_length) = shape
        total_mask = []
        for b in range(batch_size):
            potential_bias_indices = [i for i in range(seq_length - self.end_cutoff)]
            to_bias_indices = []
            for i in range(self.num_bias_positions):
                # pick the first one, since we shuffled 
                random_idx = random.sample(potential_bias_indices, 1)[0]
                to_bias_indices.append(random_idx)
                
                # update the potential bias indices
                potential_bias_indices = potential_bias_indices[:random_idx+self.bias_spacing]
            total_mask.append(to_bias_indices)
        to_mask = torch.tensor(total_mask, dtype=torch.long).to(self.device)
        mask = torch.zeros(shape, dtype=torch.float).to(self.device)
        # Use scatter_ to set mask[i, to_mask[i, :]] = 1
        mask.scatter_(1, to_mask, 1)
        
        # make it an attribute so I don't have to modify the sampling loop from the parent
        self.mask = mask
    

    def compute_bias_l2_pen(self, sampled_ids, kw_token=None):
        mask = self.mask
        bias_vec_seq = super().compute_bias_l2_pen(sampled_ids, kw_token)
        bias_vec_seq = bias_vec_seq * mask.unsqueeze(-1)
        return bias_vec_seq 

    
class DeterministicMaskingLangevin(LangevinSampler): 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_bias_positions = kwargs.get('num_bias_positions', 1)

    # get all the possible bias schedules
    def compute_all_bias_scheds(self, shape):
        (_, seq_length) = shape

        total_scheds = []
        all_combs = list(itertools.combinations(list(range(seq_length)), self.num_bias_positions))
        all_combs = [sorted(list(c)) for c in all_combs]
        return all_combs 
    
    def construct_bias_mask(self, shape, comb_batch):
        to_mask = torch.tensor(comb_batch, dtype=torch.long).to(self.device)
        mask = torch.zeros(shape, dtype=torch.float).to(self.device)
        # Use scatter_ to set mask[i, to_mask[i, :]] = 1
        mask.scatter_(1, to_mask, 1)
        # make it an attribute so I don't have to modify the sampling loop from the parent
        self.mask = mask

    def compute_bias_l2_pen(self, sampled_ids, kw_token=None):
        mask = self.mask
        bias_vec_seq = super().compute_bias_l2_pen(sampled_ids, kw_token)
        bias_vec_seq = bias_vec_seq * mask.unsqueeze(-1)
        return bias_vec_seq