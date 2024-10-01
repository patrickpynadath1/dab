from mucoco.losses import BaseLoss, register_loss

import torch 
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

import logging
import os


@register_loss("keywordclassification")
class KeyWordClassification(BaseLoss):

    def __init__(self, model, tokenizer, args):
        super().__init__() 
        
        self.model = model 
        self.tokenizer = tokenizer 
        self.args = args
        self.device = model.device
        self.topic_target = "computer"
        self.topk = args.keyword_topk
        self.topic2words = {
            "computer" : ["router", "Linux", "keyboard", "server"],
            "legal" : ["plea", "subpoena", "transcript", "bankrupt"],
            "military" : ["torpedo", "headquarters", "infantry", "battlefield"],
            "politics" : ["court", "culture", "communism", "capitalism"],
            "religion" : ["Bible", "church", "priest", "saint"],
            "science" : ["microscope", "mass", "mineral", "scientist"],
            "space" : ["meteor", "planet", "satellite", "astronaut"]
        }
        for k, v in self.topic2words.items():
            self.topic2words[k] = [torch.LongTensor(self.tokenizer.encode(w)).to(self.device) for w in v]
        # self.topic2words = {}
        # if args.topic_word_lists is not None:   
        #     vocabsize = self.model.get_input_embeddings().num_embeddings
        #     for filename in os.listdir(args.topic_word_lists):
        #         topicname = filename.split("/")[-1].split(".")[0]
        #         self.topic2words[topicname] = []
        #         for word in open(args.topic_word_lists+"/"+filename):
        #             word_tokenized = self.tokenizer.encode(" "+word)
        #             self.topic2words[topicname].append(torch.LongTensor(word_tokenized).to(self.device))
        
        self.epsilon_additive = 0#-np.log(self.topic2words[self.topic_target].sum().item())
        print("topicname", self.epsilon_additive)

    
    def compute_loss(self, batch, preds, **kwargs):
        '''
        batch: a tuple (source, prefix). If giving a prompt to the decoder, it can be specified using "prefix"
        preds: a tuple containing (predicted tokens, predicted embeddings, predicted probabilities), this is obtained through a forward pass on the optimizable target parameters (See utils/target.py)
        '''
        topic = self.topic_target
        topk = self.topk
        step = kwargs.get("step", -1)
        if len(batch) == 2:
            source, prefix = batch
        else:
            prefix = batch

        # print(topic, topk)
        pred_tokens, pred_embeds, pred_probs = preds
        batch_size = pred_embeds.size(0)
        embed_lut = self.model.get_input_embeddings()

        keyword_logits = -torch.square(torch.cdist(pred_embeds, embed_lut.weight.unsqueeze(0)))
        keyword_probs_all = F.softmax(keyword_logits, dim=-1)
        output_length = keyword_probs_all.size(1)
        losses = []
        for keyword_tokenized in self.topic2words[topic]:
            keyword_probs = keyword_probs_all.index_select(dim=-1, index=keyword_tokenized)
            ngram_length = keyword_tokenized.size(0)
            ngram_probs = [keyword_probs[:, :output_length-ngram_length+1, 0]] #first token of the ngram
            for i in range(1, ngram_length):
                ngram_probs.append(keyword_probs[:, i:output_length-ngram_length + i + 1, i])

            ngram_probs = torch.stack(ngram_probs, dim=2)
            ngram_nll = -torch.log(ngram_probs).mean(dim=-1)
            
            tau=0.1
            ngram_nll_q = F.gumbel_softmax(-ngram_nll/tau)
            loss = ngram_nll_q * ngram_nll
            loss = loss.sum(dim=-1)

            losses.append(loss)
        
        # print((-torch.stack(losses, dim=0)).topk(k=topk, dim=0))
        loss = -(-torch.stack(losses, dim=0)).topk(k=topk, dim=0)[0].mean(dim=0)
        logging_output = {
            "loss": loss.data.cpu(),
        }

        return loss, logging_output

    def compute_gold_loss(self, batch, **kwargs):
        '''
        given a discrete target output, this will compute the loss wrt to it. Useful in debugging
        '''
        source, target = batch

        batch_size=target.size(0)
    
        topic = self.topic_target
        topk = self.topk
        #input_tokens = torch.cat([bos, prefix, pred_tokens, eos], dim=1)
        embed_lut = self.model.get_input_embeddings()
        pred_embeds = embed_lut(target)
        
        # print(topic)
        keyword_logits = -torch.square(torch.cdist(pred_embeds, embed_lut.weight.unsqueeze(0)))
        keyword_probs_all = F.softmax(keyword_logits, dim=-1)
        output_length = keyword_probs_all.size(1)
        losses = []
        for keyword_tokenized in self.topic2words[topic]:
            keyword_probs = keyword_probs_all.index_select(dim=-1, index=keyword_tokenized)
            ngram_length = keyword_tokenized.size(0)
            ngram_probs = [keyword_probs[:, :output_length-ngram_length+1, 0]] #first token of the ngram
            for i in range(1, ngram_length):
                ngram_probs.append(keyword_probs[:, i:output_length-ngram_length + i + 1, i])

            ngram_probs = torch.stack(ngram_probs, dim=2)
            ngram_nll = -torch.log(ngram_probs).mean(dim=-1)
            
            tau=0.1
            ngram_nll_q = F.gumbel_softmax(-ngram_nll/tau)
            loss = ngram_nll_q * ngram_nll
            loss = loss.sum(dim=-1)

            losses.append(loss)
        
        # print(losses)
        loss = -(-torch.stack(losses, dim=0)).topk(k=topk, dim=0)[0].mean(dim=0)
        print(loss)
        # input()

        logging_output = {
            "loss": loss.data.cpu(),
        }
        return loss, logging_output   
