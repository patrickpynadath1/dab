from mucoco.losses import BaseLoss, register_loss


import torch 
import torch.nn.functional as F

@register_loss("unique")
class UniqueNessLoss(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 

        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id    
    
    def compute_loss(self, batch, preds, **kwargs):
        '''
        batch: a tuple (source, prefix). If giving a prompt to the decoder, it can be specified using "prefix"
        preds: a tuple containing (predicted tokens, predicted embeddings, predicted probabilities), this is obtained through a forward pass on the optimizable target parameters (See utils/target.py)
        '''
        prompt, prefix = batch #prompt is the real deal, prefix can be provided as an extended prompt (generated by the model autoregressively)
        pred_tokens, pred_embeds, pred_probs = preds
        pred_probs = pred_probs[0]
        batch_size = prompt.size(0)

        embed_lut = self.model.get_input_embeddings()
        # print(prefix.size(), pred_tokens.size())
        input_tokens = torch.cat([prefix, pred_tokens], dim=1)
        input_embeds = torch.cat([embed_lut(prefix), pred_embeds], dim=1)

        input_logits = -torch.square(torch.cdist(input_embeds, embed_lut.weight.unsqueeze(0)))
        input_probs = torch.softmax(input_logits, dim=-1)
        # print(input_logits.size())

        C = (torch.tril(torch.ones((input_embeds.size(1), input_embeds.size(1)))) - torch.diag(torch.ones(input_embeds.size(1)))).to(self.device)

        # print(input_tokens)
        ul = []
        for b in range(batch_size):
            ul.append(input_probs.index_select(dim=-1, index=input_tokens[b]))
        ul = torch.cat(ul, dim=0)
        # print(ul)
        # print(ul.size())
        #TODO maybe remove function words from ul last dimension
        
        # ul = 1 - ul # 1-p
        # print(ul)
        
        # print(ul)
        # print(ul.size())

        unll = torch.log(ul) #* C.unsqueeze(0)
        unll_q = (ul * C.unsqueeze(0)) /((ul * C.unsqueeze(0)).sum(dim=-1, keepdims=True) + 1e-8)
        # print(unll_q)
        # print("here")
        # print(unll.sum(dim=-1))
        # unll = unll / ((ul * C.unsqueeze(0)).sum(dim=-1, keepdims=True) + 1e-8)# (C.sum(dim=1) + 1e-6)
        unll = unll * unll_q
        # print(unll)
        unll = unll.sum(dim=-1)
        # print(unll)
        unll = unll[:, 1:]
        # print(C.sum(dim=1) + 1e-6)
        unll_qq = F.softmax(unll, dim=-1)
        # print(unll.size())
        loss = (unll_qq * unll).sum(dim=-1)
        # input(loss)
        # input(loss)
        logging_output = {
            "loss": loss.data.cpu()
        }
        return loss, logging_output

    def compute_gold_loss(self, batch, **kwargs):
        '''
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        prompt, target = batch
        batch_size = prompt.size(0)

        embed_lut = self.model.get_input_embeddings()
        input_embeds = embed_lut(target)
        input_tokens = target

        input_logits = -torch.square(torch.cdist(input_embeds, embed_lut.weight.unsqueeze(0)))
        # print(input_logits)
        input_probs = torch.softmax(input_logits, dim=-1)
        # print(input_probs)
        # print(input_tokens)
        # print(input_logits.size())

        C = (torch.tril(torch.ones((input_embeds.size(1), input_embeds.size(1)))) - torch.diag(torch.ones(input_embeds.size(1)))).to(self.device)

        #unlikelihood
        # print(input_tokens)
        ul = []
        for b in range(batch_size):
            ul.append(input_probs.index_select(dim=-1, index=input_tokens[b]))
        ul = torch.cat(ul, dim=0)
        # print(ul)
        # print(ul.size())
        #TODO maybe remove function words from ul last dimension
        
        # ul = 1 - ul # 1-p
        # print(ul)
        
        # print(ul)
        # print(ul.size())

        unll = torch.log(ul) #* C.unsqueeze(0)
        unll_q = (ul * C.unsqueeze(0)) /((ul * C.unsqueeze(0)).sum(dim=-1, keepdims=True) + 1e-8)
        # print(unll_q)
        # print("here")
        # print(unll.sum(dim=-1))
        # unll = unll / ((ul * C.unsqueeze(0)).sum(dim=-1, keepdims=True) + 1e-8)# (C.sum(dim=1) + 1e-6)
        unll = unll * unll_q
        # print(unll)
        unll = unll.sum(dim=-1)
        # print(unll)
        unll = unll[:, 1:]
        # print(C.sum(dim=1) + 1e-6)
        unll_qq = F.softmax(unll, dim=-1)
        # print(unll.size())
        loss = (unll_qq * unll).sum(dim=-1)
        # input(loss)
        # input(loss)
        logging_output = {
            "loss": loss.data.cpu()
        }

        logging_output = {"loss":loss.data.cpu()}

        return loss, logging_output   
    
