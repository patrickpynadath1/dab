import logging
import math
import os
import sys
import re
import torch
import numpy as np
import transformers
import gc
import time
import json
import random

from multiset import *

from transformers import AutoTokenizer, AutoConfig, AutoModel
from sentence_transformers import SentenceTransformer, util

from mucoco.utils import TargetProbability, TargetEmbeddings, TargetSimplex, Lambda, Optimizer, get_epsilon
import mucoco.losses as lossbuilder
import mucoco.options as options
import mucoco.utils as utils
import torch.nn.functional as F

# To control logging level for various modules used in the application:
# from here: https://github.com/huggingface/transformers/issues/3050
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

def main(args):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.ERROR,
        stream=sys.stdout,
    )
    logger = logging.getLogger("mucoco")
    logger.setLevel(logging.ERROR)
    logger.info(args)

    if args.outfile is not None:
        outf = open(args.outfile, "w")
        outallsatf = open(args.outfile + ".allsat", "w")

    # Fix seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu
    logger.info(
        "loading model(s) from {} and tokenizer(s) from {}".format(
            args.model, args.tokenizer
        )
    )

    name2tokenizer = {}
    name2model = {}
    name2config = {}
    loss2modelname = {}
    loss2tokenizer = {}
    embed_luts = []
    embed_scales = []

    betas = []
    model_paths = args.model.split(":")
    tokenizer_paths = args.tokenizer.split(":")
    cur_lr = args.lr
    args.jsonl_tokenized = args.jsonl_tokenized == "true"

    if args.model_types is not None:
        model_types = args.model_types.split(":")
    else:
        model_types = [AutoModel for _ in model_paths]

    losses = args.loss.split(":")
    if args.lossabbr is not None:
        lossabbr = args.lossabbr.split(":")
    else:
        lossabbr = [x for x in losses]

    if args.label_id is None or args.label_id == "none":
        label_ids = [1 for _ in losses]
    else:
        label_ids = [int(i) for i in args.label_id.split(":")]
    
    if args.keywords is None or args.keywords == "none":
        keywords = ["the" for _ in losses]
    elif args.keywords in ["_roc_", "_commongen_", "_commongenunique_", "_rocunique_", "_cnndm_"]:
        keywords = ["" for _ in losses] # will be different for each input
    else:
        keywords = args.keywords.split(":")
        # if len(keywords) == 1:
        #     keywords = [f"_topic_:{keywords[0]}" for _ in losses] #when keyword isn't used but topic is passed
    
    if "allsat" in args.selection_criterion: 
        # with this flag, the output which minimized the primary objective while satisfying all objectives is selected. In case all constraints are not satisfied (e.g when constraints are competing or optimization fails), this will predict the default output (Using an autoregressive decoding setup: beam search in this case)
        betas = [1.0] + [0.0 for _ in range(len(losses)-1)]
    elif (args.selection_criterion == "weighted_sum" and args.betas is not None) or args.selection_criterion == "last":
        # this setup will select the best outputs according to the weights betas for each of the losses (even though they are not satisfied)
        betas = [float(beta) for beta in args.betas.split(":")]
    else:
        raise ValueError("correct selection_criterion or betas needs to be specified")

    assert len(betas) == len(losses) and len(losses) == len(model_paths) and len(model_paths) == len(model_types) and len(betas) == len(lossabbr)
    assert np.abs(sum(betas) - 1.0) < 1e-6, f"sum of betas is {sum(betas)} != 1.0"

    prev_vocab_size = None
    vocab_size = None
    primary_vocab_size = None

    #Load the models and tokenizers
    for i, model_path in enumerate(model_paths):
        if model_path not in name2model: #making sure we are not loading the model twice in case some constraints use the same model. 
            if "#" in model_path:
                model_path_, second_model_path_ = model_path.split("#")
            else:
                model_path_ = model_path
                second_model_path_ = None
            name2tokenizer[model_path] = AutoTokenizer.from_pretrained(tokenizer_paths[i], cache_dir=args.cache_dir,  use_fast=True)
            name2config[model_path] = AutoConfig.from_pretrained(model_path_, cache_dir=args.cache_dir)

            if model_types[i] == "sentence-transformer":
                name2model[model_path] = lossbuilder.ModelWrapper(SentenceTransformer(model_path_))
            elif "Custom" in model_types[i]:
                name2model[model_path] = lossbuilder.ModelWrapper(getattr(utils, model_types[i]).from_pretrained(model_path_, config=name2config[model_path], cache_dir=args.cache_dir))
            else:
                #second model path only allowed here
                if second_model_path_ is None:
                    model_ = lossbuilder.ModelWrapper(getattr(transformers, model_types[i]).from_pretrained(model_path_, config=name2config[model_path], cache_dir=args.cache_dir))
                    name2model[model_path] = model_
                else:
                    model_ = lossbuilder.ModelWrapper(getattr(transformers, model_types[i]).from_pretrained(model_path_, config=name2config[model_path], cache_dir=args.cache_dir))
                    model2_ = lossbuilder.ModelWrapper(getattr(transformers, model_types[i]).from_pretrained(second_model_path_, config=name2config[model_path], cache_dir=args.cache_dir))
                    name2model[model_path] = (model_, model2_)

            
            if not args.show_warnings:
                # print(logging.root.manager.loggerDict)
                # input()
                if isinstance(name2model[model_path], tuple):
                    set_global_logging_level(logging.ERROR, [name2model[model_path][0].__module__])
                    set_global_logging_level(logging.ERROR, [name2model[model_path][1].__module__])
                else:
                    set_global_logging_level(logging.ERROR, [name2model[model_path].__module__])
                # logging.getLogger(name2model[model_path].__class__.__name__).setLevel(logging.ERROR) 
            
            model_.eval()
            embed_lut_ = model_.get_input_embeddings()

            if isinstance(name2model[model_path], tuple):
                model2_.eval()
            
            if isinstance(embed_lut_, torch.nn.Sequential):
                new_vocab_size = embed_lut_[0].num_embeddings
            else:
                new_vocab_size = embed_lut_.num_embeddings
            if prev_vocab_size is None:
                vocab_size=new_vocab_size
            if new_vocab_size != prev_vocab_size and prev_vocab_size is not None:
                if not args.allow_diff_vocab:
                    raise ValueError(f"all models should have the same vocabulary {new_vocab_size} != {vocab_size}")
                else:
                    logger.warning("all models don't have the same vocabulary and we are still proceeding")
            prev_vocab_size = vocab_size
        
        if args.target_tokenize_different: # for seq2seq models where target tokenizer is different than the source tokenizer
            embed_luts.append(model_.get_decoder().get_input_embeddings())
        else:
            input_embeds = model_.get_input_embeddings()
            if isinstance(input_embeds, torch.nn.Sequential):
                input_embeds = input_embeds[0]
            embed_luts.append(input_embeds)
        
        if args.target_type == "embeds":
            embed_luts[-1].requires_grad=False
        
        if i == 0:
            primary_vocab_size = vocab_size
            primary_embed_dim = embed_luts[-1].embedding_dim
        
        if getattr(model_, "get_decoder", None) is None: #this is for MarianMT models which have a weird embedding_scale parameter
            embed_scales.append(1.0)
        else:
            embed_scales.append(getattr(model_.get_decoder(), "embed_scale", 1.0))
    
    if use_cuda:
        for name, model in name2model.items():
            if isinstance(model, tuple):
                model[0].cuda()
                model[1].cuda()
            else:
                model.cuda()
        logger.info("model(s) moved to GPU")
      
    #first loss is the primary loss, others are constraints
    lossfns = []
    print(losses)
    for i, loss in enumerate(losses):
        print(loss)
        lossfns.append(lossbuilder.build_loss(loss, name2model[model_paths[i]], name2tokenizer[model_paths[i]], args))
        print(lossfns[-1])
        loss2modelname[loss] = model_paths[i]
        loss2tokenizer[loss] = name2tokenizer[model_paths[i]]
    primary_tokenizer = loss2tokenizer[losses[0]]
    primary_config = name2config[loss2modelname[losses[0]]]
    
    logger.info("tokenizer(s), model(s) and loss function(s) loaded")

    if args.model_dtype == "fp16": #while this is supported it doesn't work that well yet. Not recommended
        for name, model in name2model.items():
            model.half()
        logger.info("changed everything to fp16")

    #constraint thresholds. In the paper, we recommend to start with a high threshold value which is usually satisfied by default or easily satisfied and then decrease it gradually, otherwise weird adversarial solutions come up. This code supports different kinds of schedules for decreasing this threshold (usually just step or linear suffices). If no schedule is specified, it just remains the same as the original. 
    if args.epsilons is not None and args.epsilons != "none":
        epsilons = [float(eps) for eps in args.epsilons.split(":")]
        if args.min_epsilons is not None:
            min_epsilons = [float(eps) for eps in args.min_epsilons.split(":")]
            epsilon_warmup_steps = [int(steps) for steps in args.epsilon_warmup_steps.split(":")]
            epsilon_cooldown_steps = [int(steps) for steps in args.epsilon_cooldown_steps.split(":")]
            epsilon_decay_functions = [f for f in args.epsilon_decay_functions.split(":")]
        else:
            min_epsilons = [float(eps) for eps in args.epsilons.split(":")]
            epsilon_warmup_steps = [1 for eps in min_epsilons]
            epsilon_cooldown_steps = [2 for eps in min_epsilons]
            epsilon_decay_functions = ["none" for eps in min_epsilons]
        min_epsilons = [eps + getattr(lossfns[i+1], "epsilon_additive", 0)  for i, eps in enumerate(min_epsilons)]
    else:
        epsilons = []
        min_epsilons = []
        decay_function = []
        epsilon_warmup_steps = []
        epsilon_cooldown_steps = []
    
    # assert args.data is not None or args.additional_data is not None, "no data path has been provided"
    source_dataset = None
    target_dataset = None
    additional_dataset = None

    args.use_context = args.use_context == "true"
    print("USECONTEXTORNOT")
    print(args.use_context)
    if args.data is not None:
        data_paths = args.data.split(":")
        print(f"LENGTH OF DATAPATHS: {len(data_paths)}")
        if len(data_paths) == 1:
            source_data = data_paths[0]
            target_data = data_paths[0] #not used
            context_data = data_paths[0] # not used
            # args.use_context = False
        elif len(data_paths) == 2:
            source_data = data_paths[0]
            target_data = data_paths[1] # useful for debugging
            context_data = data_paths[1] #not used here
            # args.use_context = False
        else:
            source_data = data_paths[0]
            target_data = data_paths[1] # useful for debugging
            context_data = data_paths[2] # tsv file
    
        additional_data = args.additional_data
        if additional_data is None or additional_data == "none":
            additional_data = source_data # additional data was used in STRAP (Krishna et al 2020) when x is paraphrased to z, then the model is used to generate y in the target style. If there's no additional_data, it defaults to the source text
    elif args.additional_data is not None and additional_data != "none":
        source_data = args.additional_data
        target_data = args.additional_data
        additional_data = args.additional_data
    else:
        source_dataset = sys.stdin
        start_idx = 0
        end_idx = 1000000 # a very high number
    
    if source_dataset is None:
        logger.info("Loading the dataset ...")
        if args.datastyle == "text":
            source_dataset = [l.strip() for l in open(source_data)]
            target_dataset = [l.strip() for l in open(source_data)]
            context_dataset =[l.strip() for l in open(source_data)]
            additional_dataset = [l.strip() for l in open(source_data)] 
            import csv
            # print(f"context data {context_data}")
            # with open(context_data) as csvfile: #there can be multiple contexts, for example for paraphrasing, so we allow for a list of contexts for every input
            #     reader = csv.reader(csvfile, delimiter="\t")
            #     for row in reader:
            #         context_dataset.append(row)
            # additional_dataset = [l.strip() for l in open(additional_data)]
        elif args.datastyle == "jsonl": #for some prompts datasets
            source_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(source_data)]
            target_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(target_data)]
            additional_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(additional_data)]
            if args.jsonl_secondary_key is not None and args.jsonl_secondary_key != "none":
                source_dataset = [x[args.jsonl_secondary_key] for x in source_dataset]
                target_dataset = [x[args.jsonl_secondary_key] for x in target_dataset]
                additional_dataset = [x[args.jsonl_secondary_key] for x in additional_dataset]

            context_dataset = [None] * len(source_dataset)
            if args.use_context:
                context_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(context_data)]
                if args.jsonl_secondary_key is not None and args.jsonl_secondary_key != "none":
                    context_dataset = [x[args.jsonl_secondary_key] for x in context_dataset]
        elif args.datastyle == "single-jsonl": #one jsonl file has all the information
            print(args.jsonl_tertiary_key)
            if args.jsonl_tertiary_key == "none":
                args.jsonl_tertiary_key = args.jsonl_secondary_key
            source_dataset = [json.loads(l)[args.jsonl_primary_key] for l in open(source_data)]
            target_dataset = [json.loads(l)[args.jsonl_secondary_key] for l in open(target_data)]
            # additional_dataset = [json.loads(l)[args.jsonl_tertiary_key] for l in open(additional_data)]
            additional_dataset = [json.loads(l)[args.jsonl_secondary_key] for l in open(additional_data)]
            
            context_dataset = [None] * len(source_dataset)
            if args.use_context:
                context_dataset = [[json.loads(l)[args.jsonl_secondary_key]] for l in open(context_data)] # maybe meaningful
        start_idx = args.start_idx
        end_idx = (len(source_dataset) + args.end_idx) % len(source_dataset) + 1 # also works with negative end_idx

        logger.info("Data loaded")

    source_batch, target_batch, additional_batch, for_predicted_source_batch, predicted_batch, context_batch = [], [], [], [], [], []
    batch_size = args.batch_size # higher than 1 batch size does not work at the moment. It won't fit in a single GPU anyway 
    
    device = "cuda" if use_cuda else "cpu"
    c = 0

    losslists = [[] for _ in range(len(losses))]
    predictedlosslists = [[] for _ in range(len(losses))]
    source_primarylosslist = [] 
    # allparetosets = []
    all_stepcounts = []
    avg_time = 0

    #data loading is very simple and probably can be sped up

    if args.gold_loss_epsilons is not None and args.gold_loss_epsilons != "none":
        args.gold_loss_epsilons = args.gold_loss_epsilons.lower().split(":")
        assert len(args.gold_loss_epsilons) == len(losses)-1
    else:
        args.gold_loss_epsilons = ["false" for _ in range(len(losses)-1)]
    
    if args.custom_epsilons is not None and args.custom_epsilons != "none":
        args.custom_epsilons = args.custom_epsilons.lower().split(":")
        assert len(args.custom_epsilons) == len(losses)-1
    else:
        args.custom_epsilons = ["false" for _ in range(len(losses)-1)]

    # for source_text, target_text, additional_text in zip(source_dataset, target_dataset, additional_dataset):
    example_p = 1.0
    args.random_example = args.random_example == "true"
    if args.num_examples > 0 and target_dataset is not None:
        example_p = args.num_examples*1.0/len(source_dataset)
    print(example_p, args.random_example)
    print(start_idx, end_idx)
    print("LENGTH OF THE DATASET", len(context_dataset))
    prev_garbage = None
    cur_garbage = Multiset()
    for text_id, source_text in enumerate(source_dataset):
        # end_idx = 1000
        # if text_id < start_idx or text_id > end_idx:
        #     continue

        if args.num_examples > 0 and c > 0 and c == args.num_examples: #stop after processing num_examples if it is set 
            print(f"done {c}")
            break
        
        # do_this_example = np.random.rand() <= example_p
        # if not do_this_example:
        #     continue
        
        # print(text_id, "doing it! do_this_example")

        c += 1

        new_kweight = args.kweight
        if target_dataset is not None:
            target_text = target_dataset[text_id]
            additional_text = additional_dataset[text_id]
            context_texts = context_dataset[text_id]
            # print(context_texts)
            # input()
        else:
            args.jsonl_tokenized = False
            items = source_text.split("::")
            source_text = items[0].rstrip()
            target_text = items[1].rstrip()
            if target_text == "-":
                args.init = "zeros"
            elif target_text == "--":
                args.init = "target"
            else:
                args.init = "targettarget"
                print("aaaaaaaaaaaaaaaa")
            additional_text = items[2]

            if len(items) > 3:
                args.max_output_length = int(items[3])
                args.max_length = int(items[3])
            if len(items) > 4:
                args.use_context = True
                context_texts = [items[4]].rstrip()                               
            else:
                args.use_context = False
                context_texts = []

            if len(items) > 5:
                new_kweight = float(items[5])
            # if len(items) > 4:
            #     target_text = items[4].rstrip()
            
            print(args.use_context, context_texts)
        topics = ["computer", "legal", "military", "religion", "politics", "science", "space"]
        if args.keywords == "_roc_":
            keywords = ["none"] + additional_text.split(", ")
            # input(keywords)
        # edits for BOLT keyword gen here
        elif args.keywords == "computer": 
            keywords = ["none","router", "Linux", "keyboard", "server", "none"]
            main_keywords =["router", "Linux", "keyboard", "server"]
        
        elif args.keywords == "legal": 
            keywords = ["none","plea", "subpoena", "transcript", "bankrupt", "none"]
            main_keywords = ["plea", "subpoena", "transcript", "bankrupt"]
        
        elif args.keywords == "military": 
            keywords = ["none","torpedo", "headquarters", "infantry", "battlefield", "none"]
            main_keywords = ["torpedo", "headquarters", "infantry", "battlefield"]
        
        elif args.keywords == "religion": 
            keywords = ["none","Bible", "church", "priest", "saint", "none"]
            main_keywords = ["Bible", "church", "priest", "saint"]
        
        elif args.keywords == "politics": 
            keywords = ["none","court", "culture", "communism", "capitalism", "none"]
            main_keywords = ["court", "culture", "communism", "capitalism"]
        
        elif args.keywords == "science": 
            keywords = ["none","microscope", "mass", "mineral", "scientist", "none"]
            main_keywords = ["microscope", "mass", "mineral", "scientist"]
        
        elif args.keywords == "space": 
            keywords = ["none","meteor", "planet", "satellite", "astronaut", "none"]
            main_keywords = ["meteor", "planet", "satellite", "astronaut"]
        elif args.keywords == "_rocunique_":
            keywords = ["none"] + additional_text.split(", ") + ['none']
        elif args.keywords == "_rocunique_nn":
            keywords = ["none"] + additional_text.split(", ") + ['none'] + ["\n", "\n\n"]
            # input(keywords)
        elif args.keywords == "_commongen_":
            print(additional_text)
            if args.datastyle=="text":
                keywords = ["none"] + json.loads(additional_text)['concept_set'].split("#")
            else:
                keywords = ["none"] + additional_text.split("#")
        elif "_commongenunique_" in args.keywords:
            if args.datastyle=="text":
                keywords = ["none"] + json.loads(additional_text)['concept_set'].split("#") + ["none"]
            else:
                print(additional_text)
                keywords = ["none"] + additional_text.split("#") + ["none"]
        elif "_commongenmorphounique_" in args.keywords:
            keywords = ["none"] + ["#".join(words) for words in eval(additional_text)] + ["none"]
        elif "commongenmorpho" in args.keywords:
            keywords = ["none"] + ["#".join(words) for words in eval(additional_text)]
        elif "iate" in args.keywords:
            keywords = [word for i, word in enumerate(additional_text.strip().split("\t")[2:]) if i%2 == 1] #words at index 1, 3, 5 are translations in target
            keywords = ["none"] + keywords
            # if args.datastyle=="text":
            #     keywords = ["none"] + json.loads(additional_text)['concept_set'].split("#") + ["none"]
            # else:
            #     print(additional_text)
            #     keywords = ["none"] + additional_text.split("#") + ["none"]
        # elif args.keywords == "_commongenunique_n":
        #     if args.datastyle=="text":
        #         keywords = ["none"] + json.loads(additional_text)['concept_set'].split("#") + ["none"] + ["\n"]
        #     else:
        #         print(additional_text)
        #         keywords = ["none"] + additional_text.split("#") + ["none"] + ["\n"] #blacklist \n
        #     print(keywords)
        # elif args.keywords == "_commongenunique_nn":
        #     if args.datastyle=="text":
        #         keywords = ["none"] + json.loads(additional_text)['concept_set'].split("#") + ["none"] + ["\n", "\n\n"]
        #     else:
        #         print(additional_text)
        #         keywords = ["none"] + additional_text.split("#") + ["none"] + ["\n", "\n\n"] #blacklist \n
        #     print(keywords)
        elif args.keywords == "_cnndm_":
            # print(additional_text)
            if len(additional_text) == 0:
                additional_text = ["the"]
            random.shuffle(additional_text)
            # keywords = ["none", "the"] # this is defined later. 
            # else:

        if "_nn" in args.keywords:
            keywords = keywords + ["\n", "\n\n"]
            # args.keywords = args.keywords.split("_")[0]
        elif "_n" in args.keywords:
            keywords = keywords + ["\n"]
        elif "_pad" in args.keywords:
            keywords = keywords + ["</s>"]
            # input(keywords)

        
        early_skip="n"
        if args.debug:
            early_skip = input(f"skip this example? {source_text} [yes(y)/maybe(m)/no(n)]")
            if early_skip == "y":
                continue

        if not args.jsonl_tokenized:
            if source_text == "":
                source_text = primary_tokenizer.bos_token
                source_indices = torch.LongTensor([[primary_tokenizer.bos_token_id]]).to(device)
            elif "commongen" in args.keywords:
                source_indices = primary_tokenizer.encode(" "+source_text, return_tensors="pt", truncation=True, max_length=getattr(primary_config, "max_position_embeddings", None)).to(device)
                bos = torch.LongTensor([[primary_tokenizer.bos_token_id]]).to(device)
                source_indices = torch.cat([source_indices,bos], dim=1)
            else:
                source_indices = primary_tokenizer.encode(" "+source_text, return_tensors="pt", truncation=True, max_length=getattr(primary_config, "max_position_embeddings", None)).to(device)
            source_indices_write = source_indices[0].tolist()
            # if source_indices
            if isinstance(additional_text, list):
                additional_indices = primary_tokenizer.encode(additional_text[0], return_tensors="pt", add_special_tokens=False).to(device)
            else:
                additional_indices = primary_tokenizer.encode(additional_text, return_tensors="pt", add_special_tokens=False).to(device)
            
            eos_token_id = primary_tokenizer.eos_token_id
            bos_token_id = primary_tokenizer.bos_token_id
            context_indices = None
            # print(args.target_tokenize_different)
            # input("sfdsfsdf")
            if args.target_tokenize_different:
                with primary_tokenizer.as_target_tokenizer():
                    eos_token_id=primary_tokenizer.eos_token_id
                    bos_token_id = primary_tokenizer.bos_token_id
                    if args.use_context:
                        context_indices = primary_tokenizer.encode(context_texts[0], return_tensors="pt", add_special_tokens=False).to(device).unsqueeze(1)
            elif args.use_context:
                context_indices = primary_tokenizer.encode(context_texts[0], return_tensors="pt", add_special_tokens=False).to(device).unsqueeze(1)

            if not args.target_tokenize_different and "Seq2SeqLM" in model_paths[0]:
                logger.warning("you are using a seq2seq model for your primary loss but not tokenizing the target sentences with a different target tokenizer.")

            #for_predicted_source_indices are used to compute the primary loss wrt source as target. Useful for debugging style transfer models. 
            if args.target_tokenize_different:
                with primary_tokenizer.as_target_tokenizer():
                    for_predicted_source_indices = primary_tokenizer.encode(source_text, return_tensors="pt").to(device)
                    target_indices = primary_tokenizer.encode(target_text, return_tensors="pt", add_special_tokens=False).to(device)
            else:
                for_predicted_source_indices = source_indices
                target_indices = primary_tokenizer.encode(target_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        else:
            source_indices_write = source_text # to write to file
            source_indices = source_text
            target_indices = target_text
            additional_indices = additional_text
            context_indices = context_texts
            if len(source_indices) == 0:
                source_indices.append(primary_tokenizer.bos_token_id)

            source_indices = torch.LongTensor([source_indices]).to(device)
            additional_indices = torch.LongTensor([additional_indices]).to(device)
                        
            #unused
            context_indices = None
            if args.use_context:
                context_indices = torch.LongTensor([context_indices]).to(device).to(device).unsqueeze(1)
            #end unused

            #for_predicted_source_indices are used to compute the primary loss wrt source as target. Useful for debugging style transfer models. 
            for_predicted_source_indices = source_indices
            target_indices = torch.LongTensor([target_indices]).to(device)

            bos_token_id = primary_tokenizer.bos_token_id
            eos_token_id = primary_tokenizer.eos_token_id
            if args.target_tokenize_different:
                with primary_tokenizer.as_target_tokenizer():
                    bos_token_id = primary_tokenizer.bos_token_id
                    eos_token_id = primary_tokenizer.eos_token_id
            
            source_text = primary_tokenizer.decode(source_indices[0].tolist())

        source_batch.append(source_indices)
        target_batch.append(target_indices)
        for_predicted_source_batch.append(for_predicted_source_indices)
        additional_batch.append(additional_indices)
        context_batch.append(context_indices)
        print(source_text)
        if len(source_batch) == batch_size: #this is just one for now, greater than 1 batch size will not work

            source_batch = torch.cat(source_batch, dim=0).to(device)
            target_batch = torch.cat(target_batch, dim=0).to(device)
            additional_batch = torch.cat(additional_batch, dim=0).to(device)
            for_predicted_source_batch = torch.cat(for_predicted_source_batch, dim=0).to(device)  
            
            # print("what", args.use_context)
            if args.use_context:
                context_batch = torch.cat(context_batch, dim=0).to(device)
                print(context_batch)

            # generating Autoregressive samples
            predicted_batches = [] #each sample x restart becomes a tensor
            for batchidx in range(source_batch.size(0)): #batch size is 1
                with torch.no_grad():
                    starttime = time.time()
                    AR_predicted_all =\
                        lossfns[0].generate(
                            input_ids=source_batch[batchidx].unsqueeze(0),
                            additional_ids=additional_batch[batchidx].unsqueeze(0),
                            num_beams=args.beam_size,
                            num_return_sequences=(args.restarts + 1)*args.num_samples) 

                    # AR_predicted_indices_all = []
                    AR_prediction_all = []
                    for sample_idx in range(len(AR_predicted_all)):
                        AR_predicted_indices =\
                            clean_output(AR_predicted_all[sample_idx].tolist(),
                                eos_token_id=eos_token_id,
                                return_tensors=True, allow_first_eos=losses[0] == "bart", remove_first = losses[0] == "marianmt",
                                skip_special_tokens=[bos_token_id, eos_token_id])
                        # AR_predicted_indices_all.append(AR_predicted_indices)

                        if args.target_tokenize_different:
                            with primary_tokenizer.as_target_tokenizer():
                                AR_prediction = primary_tokenizer.decode(AR_predicted_indices[0].tolist())
                        else:
                            AR_prediction = primary_tokenizer.decode(AR_predicted_indices[0].tolist())
                        AR_prediction_all.append(AR_prediction)
                        
                        # predicted_batch.append(AR_predicted_indices)
                        predicted_batches.append(AR_predicted_indices.to(device))
                    if args.time:
                        print(time.time()-starttime)

            
            broken_skip = False
            
            for sample_idx in range(args.num_samples):
                
                if args.keywords == "_cnndm_":
                    if sample_idx < len(additional_text):
                        keywords = ["none", additional_text[sample_idx]]
                        print("current keyword:", keywords[-1])
                    elif len(additional_text) == 0:
                        keywords = ["none", "the"]
                    else:
                        break

                for restart_idx in range(args.restarts + 1): # restart the optimization if the constraints are not satisfied
                    predicted_batch = predicted_batches[(sample_idx * (args.restarts + 1) + restart_idx) % len(predicted_batches)]
                    AR_prediction = AR_prediction_all[(sample_idx * (args.restarts + 1) + restart_idx) % len(predicted_batches)]

                    ##TODO: in case of always_mucoco=false and num_restarts > 0, comb through the restarts and skip if constraints are satisfied

                    skip=False
                    predicted_allsat=False
                    lengthwise_best_prediction = [None] * batch_size

                    if args.debug:
                        print("AR output:", source_text, additional_text, predicted_batch)

                    # losses of the autoregressive output: we should perform atleast as well as this. If we don't, we predict this output
                    # Also, if the autoregressive output already satisfies the constraints, we skip mucoco unless, args.always_mucoco is true
                    predicted_labels = {}
                    total_predicted_loss = 0.0
                    predicted_allsat=True
                    predictedlosses = []
                    
                    for lossid in range(len(losses)):
                        predicted_loss, predicted_lo =\
                            lossfns[lossid].compute_gold_loss(
                                (source_batch, predicted_batch), 
                                additional_batch=additional_batch, 
                                context_batch=context_batch,
                                use_context=args.use_context,
                                label_id=label_ids[lossid],
                                keyword=keywords[lossid],
                                kweight=new_kweight)
                        
                        if lossid > 0 and args.custom_epsilons[lossid-1] == "true": #use a custom epsilon value defined by the loss class (used with ngram-inverse)
                            min_epsilons[lossid - 1] = getattr(lossfns[lossid], "epsilon", 0)
                            epsilons[lossid - 1] = 100#(lossfns[lossid], "epsilon", 0)
                        
                        predictedlosses.append(predicted_loss.data.cpu())
                        del predicted_loss
                        predicted_loss = predictedlosses[-1].sum().item()
                        total_predicted_loss += betas[lossid] * predicted_loss

                        if lossid > 0:
                            predicted_allsat = predicted_allsat and (predicted_loss <= min_epsilons[lossid-1])
                        
                        if "label_prediction" in predicted_lo:
                            predicted_labels[lossid] = predicted_lo['label_prediction']
                        else:
                            predicted_labels[lossid] = "NA"
                        
                        if lossid > 0 and args.gold_loss_epsilons[lossid-1] == "true": #use the predicted loss as the threshold, mucoco has to beat it then
                            min_epsilons[lossid - 1] = predicted_loss + getattr(lossfns[lossid], "epsilon_additive", 0)
                            epsilons[lossid - 1] = predicted_loss + getattr(lossfns[lossid], "epsilon_additive", 0) ##TODO check 
                        
                    predictedlosslists.append(predictedlosses)
                    
                    if args.only_mucoco == "false":
                        lengthwise_best_prediction = [(AR_prediction, total_predicted_loss, predicted_allsat, predicted_batch[0].tolist(), -1)]
                    
                    if args.debug and lengthwise_best_prediction[0] is not None and lengthwise_best_prediction[0][2]:
                        print("lbp", lengthwise_best_prediction)
                        print(predictedlosses, min_epsilons)      
                        # input()

                    skip = predicted_allsat
                    definite_skip = False
                    ask_skip = ""

                    if args.debug and early_skip=="m": 
                        print(f"new example: {source_text}\nautoregressive output: {AR_prediction}")
                        for lossid in range(len(losses)):
                            print(f"{lossabbr[lossid]} for desired label_id({label_ids[lossid]}): {predictedlosslists[-1][lossid]}; predicted label: {predicted_labels[lossid]}")
                        if predicted_allsat:
                            print(f"autoregressive output already satisfies the constraints")
                        ask_skip = input(f"skip this example? [y/n]")
                        definite_skip = ask_skip == "y"

                    elif skip and predicted_allsat and (args.always_mucoco == "false"):
                        definite_skip = True

                    if args.debug:
                        print('definite_skip', definite_skip, skip, predicted_allsat, args.always_mucoco)
                    
                    if not definite_skip:
                        # print(args.max_length)
                        # input("sdfsdfd")
                        # if (args.max_length is None or args.max_length == -1) and args.init not in ["source", "target"]: 
                        if (args.length_diff is not None and args.length_diff != "none") and args.init not in ["source", "target"]: 
                            #since we don't know the about length, we search in a (-length_diff, length_diff) window and predict the best performing one.
                            predicted_length = predicted_batch.size(1)
                            length_range = [predicted_length + int(diff) for diff in args.length_diff.split(":")]
                            length_range = [x for x in length_range if x <= args.max_allowed_length and x >= 1]
                            if len(length_range) == 0:
                                length_range = [args.max_allowed_length]
                            length_range = sorted(list(set(length_range)))
                            # print(length_range)
                            # input("sdfdsf")
                        elif args.init == "targettarget":
                            length_range = [target_batch.size(1)]
                        elif args.init == "target":
                            length_range = [predicted_batch.size(1)]
                        elif args.init == "source":
                            length_range = [source.size(1)]
                        else: 
                            #another way to use this approach is train models which also compute loss on <pad> token and then predict the entire sentence including pad, it has shown to work in some of our experiments
                            length_range = [args.max_length]  
                    
                        for sent_length_ in length_range:
                            # print(sent_length_)
                            # prefix_length is used to indicate if instead of predicting the entire sentence via optimization, we want to fix a prefix (of specified length) and predict the remaining suffix. We use part of the beam search prediction as the prefix during debugging. 
                            if args.prefix_length > 0:
                                sent_length = sent_length_ - args.prefix_length
                                target_prefix = predicted_batch[:, :args.prefix_length]
                            else:
                                sent_length = sent_length_
                                target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                            
                            if sent_length <= 0:
                                continue
                            if sent_length > args.max_allowed_length:
                                #max_allowed_length is just to make sure things don't go out of memory,
                                old_l = sent_length
                                sent_length = args.max_allowed_length
                                print(f"changed output length to {sent_length} from {old_l} to avoid GPU overflow. This is a temporary solution")
                            else:
                                print("predicting a sentence length: ", sent_length)
                                
                            if args.target_type == "simplex": # use V sized real vector for each token and apply softmax before output
                                init_value = None
                                break_after=False
                                outputs = TargetSimplex(
                                    vocabsize=primary_vocab_size,
                                    sent_length=sent_length,
                                    batch_size=batch_size,
                                    device=device,
                                    temperature=args.decode_temperature,
                                    st=args.st,
                                    init_value=source_batch[:,1:-1] if args.init == "source" else None,
                                    random_init=args.init == "random",
                                    do_sample=args.expgd_do_sample,
                                    top_p=args.expgd_top_p,
                                    top_k=args.expgd_top_k,
                                    embed_scales=embed_scales,
                                    sampling_strategy=args.sampling_strategy,
                                    sampling_strategy_k=args.sampling_strategy
                                )
                            elif args.target_type == "probs": # use V sized vector which sums to one for each token and apply softmax before output
                                init_value = None
                                break_after=False
                                if args.init == "source": #initialize the target with the source
                                    init_value = source_batch
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after=True
                                    # print(source_batch, init_value, sent_length, init_value)
                                elif args.init == "target": #initialize the target with the autoregressive output
                                    init_value = target_batch
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after=True
                                    # print(source_batch, init_value)
                                outputs = TargetProbability(
                                    vocabsize=primary_vocab_size,
                                    sent_length=sent_length,
                                    batch_size=batch_size,
                                    device=device,
                                    st=args.st,
                                    init_value=init_value,
                                    random_init=args.init == "random",
                                    do_sample=args.expgd_do_sample,
                                    top_p=args.expgd_top_p,
                                    top_k=args.expgd_top_k,
                                    embed_scales=embed_scales,
                                    max_steps=args.optim_steps
                                )
                            elif args.target_type == "embeds":
                                init_value = None
                                break_after=False
                                if args.init == "source": #initialize the target with the source
                                    init_value = embed_luts[0](source_batch)
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after=True
                                    # print(source_batch, init_value, sent_length, init_value)
                                elif args.init == "targettarget": #initialize the target with given target
                                    init_value = embed_luts[0](target_batch)
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after=True 
                                    print(predicted_batch.size())   
                                    print(sent_length)
                                elif args.init == "target": #initialize the target with the autoregressive output
                                    init_value = embed_luts[0](predicted_batch)
                                    target_prefix = torch.empty((source_indices.size(0), 0)).long().to(device)
                                    sent_length = init_value.size(1)
                                    break_after=True 
                                    print(predicted_batch.size())   
                                    print(sent_length)

                                elif args.init == "random_vocab":
                                    random_indices = torch.multinomial(torch.ones(primary_vocab_size,)/primary_vocab_size, num_samples=batch_size*sent_length, replacement=True).view(batch_size, sent_length).to(device)
                                    init_value = embed_luts[0](random_indices)
                                elif args.init == "embedgd-zeros":
                                    if args.target_tokenize_different:
                                        with primary_tokenizer.as_target_tokenizer():
                                            indices = torch.empty((batch_size, sent_length)).long().fill_(primary_tokenizer.eos_token_id).to(device)
                                    else:
                                        # indices = torch.empty((batch_size, sent_length)).long().fill_(primary_tokenizer.eos_token_id).to(device)
                                        indices = torch.empty((batch_size, sent_length)).long().fill_(198).to(device)
                                    # print(primary_tokenizer.decode(indices[0]))
                                    init_value = embed_luts[0](indices)
                                elif args.init == "zeros":
                                    indices = torch.zeros((batch_size, sent_length)).long().to(device)
                                    init_value = embed_luts[0](indices)

                                
                                final_bias = None
                                if args.final_bias:
                                    final_bias = lossfns[0].model.final_logits_bias

                                # print(sent_length)
                                outputs = TargetEmbeddings(
                                    embed_dim=primary_embed_dim,
                                    embed_lut=embed_luts[0],
                                    sent_length=sent_length,
                                    batch_size=batch_size,
                                    device=device,
                                    st=args.st,
                                    init_value=init_value,
                                    random_init=args.init == "random",
                                    sampling_strategy=args.sampling_strategy,
                                    sampling_strategy_k=args.sampling_strategy_k,
                                    embed_scales=embed_scales,
                                    metric=args.metric,
                                    same_embed=args.same_embeds,
                                    final_bias=final_bias,
                                    eos_token_id=primary_tokenizer.eos_token_id
                                )
                            else:
                                raise ValueError("Wrong target_type")

                            if len(losses) > 1:
                                lambda_ = Lambda(count=len(epsilons))
                                if use_cuda:
                                    lambda_.cuda()

                            optimizer = Optimizer.from_opt(outputs, args)
                            cur_lr = args.lr
                            # print(optimizer._optimizer.param_groups)
                            # input()
                            if len(losses) > 1:
                                old_optim = args.optim
                                args.optim = "gradascent"
                                old_lr = args.lr
                                args.lr = args.lambda_lr
                                optimizer_lambda = Optimizer.from_opt(lambda_, args)
                                args.optim = old_optim
                                args.lr = old_lr

                            best_loss = [None] * batch_size
                            best_allsat = [False] * batch_size
                            best_repeat_count = [0] * batch_size
                            best_losses = [[None] * batch_size for _ in range(len(losses))]
                            best_step = -100
                            
                            best_pred_tokens = [None] * batch_size
                            best_prediction_set = [set() for _ in range(batch_size)]
                            best_pred_probs = [None] * batch_size
                            best_index = [-1 for i in range(batch_size)]
                            
                            scaler = None
                            if args.model_dtype == "fp16" and args.fp16_source == "pytorch":
                                scaler = torch.cuda.amp.GradScaler()
                        
                            for lossid, lossname in enumerate(losses):
                                losslists[lossid].append([])

                            broken = False
                            prev_loss = None
                            dynamic_lambda_update_prev_loss = None
                            same_loss_count = 0
                            dynamic_loss_update_same_loss_count = 0
                            starttime = time.time()
                            repeat_counts = [0] * batch_size

                            update_lr_condition = "none"
                            for step in range(args.optim_steps):
                                try:
                                    with torch.cuda.amp.autocast(enabled=False):
                                        losses_for_backward = []
                                        logging_outputs = []

                                        # print(optimizer.new_predictions)
                            
                                        pred_embeds, pred_tokens, pred_probs = outputs.forward_multiple(embed_luts, new_predictions=getattr(optimizer._optimizer, "new_predictions", None))  # forward

                                        if not args.time and args.debug:
                                            def get_sent(tokens, tokenizer):
                                                batch = []
                                                if args.target_tokenize_different:
                                                    with tokenizer.as_target_tokenizer():
                                                        for toks in tokens:
                                                            batch.append(tokenizer.decode(clean_output(toks.tolist(), -1, allow_first_eos=losses[0] == "bart")))
                                                else:
                                                    for toks in tokens:
                                                        batch.append(tokenizer.decode(clean_output(toks.tolist(), -1, allow_first_eos=losses[0] == "bart")))
                                                return batch

                                            target_sents = get_sent(torch.cat([target_prefix, pred_tokens], dim=1), primary_tokenizer)
                                            print(target_sents, end="\n")
                                        
                                        original_preds = None
                                        if len(pred_embeds) > 1:
                                            original_preds = pred_embeds[1]

                                        optimizer.zero_grad(set_to_none=True)
                                        outputs.zero_grad()
                                        if len(losses) > 1:
                                            optimizer_lambda.zero_grad(set_to_none=True)
                                            lambda_.zero_grad()

                                        for model in name2model.values():
                                            if isinstance(model, tuple):
                                                model[0].zero_grad(set_to_none=True)
                                                model[1].zero_grad(set_to_none=True)
                                            else:
                                                model.zero_grad(set_to_none=True)

                                        # print("what", args.use_context)
                                        for lossid, lossname in enumerate(losses):
                                            lossvalue, logging_output =\
                                                lossfns[lossid].compute_loss(
                                                    [source_batch, target_prefix], 
                                                    [pred_tokens, pred_embeds[0][lossid], pred_probs], 
                                                    additional_batch=additional_batch, 
                                                    context_batch=context_batch,
                                                    use_context=args.use_context,
                                                    embed_scale=embed_scales[lossid], 
                                                    label_id=label_ids[lossid],
                                                    keyword=keywords[lossid],
                                                    original_preds=original_preds,
                                                    kweight=new_kweight,
                                                    step=step
                                                )

                                            losslists[lossid][-1].append(lossvalue.sum().item())  #for logging
                                            losses_for_backward.append(lossvalue)  # for backward
                                            logging_outputs.append(logging_output)
                                                                            
                                        if args.linear_scale == "true": # no lagragian, plain old linear sum
                                            # 
                                            # grads = []
                                            # if args.debug and args.debug_gradients == "true":
                                            #     for sid in range(len(losses_for_backward)):
                                            #         optimizer.backward(losses_for_backward[sid], retain_graph=True, scaler=scaler)
                                            #         grad = []
                                            #         for p in outputs.parameters():
                                            #             grad.append(p.grad.data)
                                            #             param_norm = p.grad.data.norm(2, -1).sum(dim=0)
                                            #             print(sid, "for theta", param_norm)
                                            #         grads.append(grad[0])
                                            #         optimizer.zero_grad(set_to_none=True)
                                            #         outputs.zero_grad(set_to_none=True)
                                            #         for modelname in loss2modelname.values():
                                            #             name2model[modelname].zero_grad(set_to_none=True) 
                                            #     graddot = (grads[0] * grads[1]).sum(dim=-1)
                                            #     print(graddot)
                                            #     grads0norm = torch.nn.functional.normalize(grads[0], p=2, dim=-1)
                                            #     grads1norm = torch.nn.functional.normalize(grads[1], p=2, dim=-1)
                                            #     print((grads0norm * grads1norm).sum(dim=-1))
                                                # input()
                                            # else:
                                            # total_loss = betas[0] * losses_for_backward[0]
                                            total_loss = 0
                                            cur_epsilons = [] # just for avoiding syntax errors, epsilons are useless in this setting
                                            for sid in range(len(losses_for_backward)):
                                                total_loss = total_loss + betas[sid] * losses_for_backward[sid]
                                                cur_epsilons.append(0.0)
                                            
                                            total_batchloss = total_loss.sum()
                                            optimizer.backward(total_batchloss, retain_graph=False, scaler=scaler)
                                        else:
                                            total_loss = 0.0
                                            total_loss = losses_for_backward[0]
                                            # total_loss_for_lambda = 0.0
                                            cur_epsilons = []
                                            # print(total_loss.item(), end=", ")
                                            
                                            if args.debug and args.debug_gradients == "true":
                                                optimizer.backward(total_loss.sum(), retain_graph=True, scaler=scaler)
                                                g = [list(outputs.parameters())[0].grad.data]
                                                optimizer.zero_grad(set_to_none=True)
                                                for modelname in loss2modelname.values():
                                                    name2model[modelname].zero_grad(set_to_none=True) 

                                            constraint_values = []
                                            for sid in range(1, len(losses_for_backward)): #the secondary losses or constraints
                                                
                                                cur_epsilon = get_epsilon(step, epsilons[sid-1], min_epsilons[sid-1], epsilon_warmup_steps[sid-1], epsilon_cooldown_steps[sid-1], epsilon_decay_functions[sid-1])
                                                # print(epsilons[sid-1], cur_epsilon)
                                                # input()
                                                
                                                constraint_value = (cur_epsilon - losses_for_backward[sid]).item()
                                                damp = args.dampness * constraint_value
                                                mask = lambda_.get_mask(sid-1, damp)
                                                
                                                # closs_for_theta = - lambda_.get_loss(sid - 1, damp * mask, (cur_epsilon - losses_for_backward[sid]))
                                                total_loss = total_loss - lambda_.get_loss(sid - 1, damp * mask, (cur_epsilon - losses_for_backward[sid]))
                                                # print(damp, mask, lambda_.get_loss(sid - 1, damp * mask, (cur_epsilon - losses_for_backward[sid])))

                                                if args.debug and args.debug_gradients == "true":
                                                    optimizer.backward(closs_for_theta.sum(), retain_graph=True, scaler=scaler)
                                                    g.append(list(outputs.parameters())[0].grad.data)
                                                    optimizer.zero_grad(set_to_none=True)
                                                    for modelname in loss2modelname.values():
                                                        name2model[modelname].zero_grad(set_to_none=True) 

                                                
                                                cur_epsilons.append(cur_epsilon)                             
                                                constraint_values.append(constraint_value)
                                        
                                            total_batchloss = total_loss.sum()

                                            # t = torch.cuda.get_device_properties(0).total_memory
                                            # r = torch.cuda.memory_reserved(0)
                                            # a = torch.cuda.memory_allocated(0)
                                            # f = r-a  # free inside reserved

                                            # print("###########MEMORY BeforeBACKWARD:", t, r, a, f)

                                            optimizer.backward(total_batchloss, retain_graph=False, scaler=scaler)

                                            # t = torch.cuda.get_device_properties(0).total_memory
                                            # r = torch.cuda.memory_reserved(0)
                                            # a = torch.cuda.memory_allocated(0)
                                            # f = r-a  # free inside reserved

                                            # print("###########MEMORY AfterBACKWARD:", t, r, a, f)

                                        if args.debug and args.debug_gradients == "true":
                                            totals = 0
                                            for i in range(len(g)):
                                                totals = g[i] + totals
                                                for j in range(len(g)):
                                                    dot = (g[i] * g[j]).sum(dim=-1)
                                                    s = g[i] + g[j]
                                                    cosine = dot/(torch.norm(g[i], dim=-1, p=2) * torch.norm(g[j], dim=-1, p=2))
                                                    snorm = s.data.norm(2, -1)
                                                    print(i, j, dot, cosine, s)
                                            print("total sum norm", totals.data.norm(2, -1))

                                            total_norm = 0
                                            gi=0
                                            for p in outputs.parameters():
                                                gi+=1
                                                param_norm = p.grad.data.norm(2, -1).sum(dim=0)
                                                # print(p.dtype)
                                                print("for theta", param_norm)
                                            for p in lambda_.parameters():
                                                print("for lambda", p.grad)

                                            if (step % args.lambda_update == 0):
                                                input()
                                    
                                    if logging_outputs[0].get('entropy', None) is not None:
                                        optimizer.step(scaler=scaler, entropy=logging_outputs[0].get('entropy', None))
                                    else:
                                        optimizer.step(scaler=scaler)
                                    
                                    # new_update_lr_condition = "none"
                                    if args.linear_scale != "true" and  len(losses) > 1:
                                        sats = torch.Tensor(constraint_values).ge(0.).to(device)
                                        update_lambda_condition = (step % args.lambda_update == 0)
                                        lambda_mask = float(update_lambda_condition) * torch.ones_like(sats)
                                        
                                        lambda_mask += (1-sats.float()) * (lambda_.is_zero())
                                        # if not sats.all() and (lambda_.any_zero()):
                                        #     print("funky new update")
                                        #     update_lambda_condition = True
                                        #     lambda_mask = torch.ones_like(sats)
                                        # lambda_mask += sats.float()

                                        # if args.linear_scale != "true" and  len(losses) > 1 and args.dynamic_lambda_update:# and "roc" in args.keywords:
                                        #     lambda_mask = (1 - sats.float())
                                        # if step > args.lambda_update:
                                        
                                    
                                    # total_batchlossitem = total_batchloss.item()
                                    total_batchlossitem = losses_for_backward[0].item()
                                    # if dynamic_lambda_update_prev_loss is not None:
                                        # print(abs(total_batchlossitem - dynamic_lambda_update_prev_loss))
                                    if dynamic_lambda_update_prev_loss is not None and abs(total_batchlossitem - dynamic_lambda_update_prev_loss) <= 1e-6:
                                        repeat_counts[0] += 1
                                        if args.linear_scale != "true" and  len(losses) > 1 and args.dynamic_lambda_update:
                                            lambda_mask = (1 - sats.float())
                                            
                                            # if "commongen" in args.keywords:
                                                # for lossid in range(len(lossfns)):
                                                    # print(type(lossfns[lossid]))
                                                    # if isinstance(lossfns[lossid], "mucoco.losses.ngrams-l2.KeywordL2Loss")
                                                    # lossfns[lossid].fix_q()

                                            # print("what now", total_batchlossitem, dynamic_lambda_update_prev_loss, constraint_values, sats.float())
                                            # if sats.all(): #constraints are satisfied
                                            #     update_lambda_condition = False
                                            #     print("constraints are satisfied and output is not changing, lambdas will not update!")
                                            # else:
                                            #     update_lambda_condition = True

                                        if args.dynamic_lr_update and best_allsat[0] is not None and best_allsat[0]:
                                            update_lr_condition = "increase"
                                            cur_lr = optimizer._optimizer.update_lr(min(cur_lr + args.lr_update_size, args.max_lr))
                                    else:
                                        # if update_lr_condition == "increase":
                                        #     #reset learning rate after a change has been triggered
                                        #     cur_lr = optimizer._optimizer.update_lr(max(cur_lr - args.lr_update_size, args.lr))
                                        #     update_lr_condition = "decrease"
                                        # elif update_lr_condition == "decrease":
                                        #     #reset learning rate after a change has been triggered
                                        #     cur_lr = optimizer._optimizer.update_lr(max(cur_lr - args.lr_update_size, args.lr))
                                        # cur_lr = args.lr
                                        repeat_counts[0] = 1
                                    # print(repeat_counts)
                                    
                                    dynamic_lambda_update_prev_loss = total_batchlossitem

                                    # if update_lr_condition == "increase":
                                        

                                    if args.linear_scale != "true" and len(losses) > 1:
                                        # print(lambda_mask, repeat_counts)
                                        # print([p.grad for p in lambda_.parameters()])
                                        # print(step, lambda_().tolist(), lambda_mask, )
                                        # print(lambda_mask)
                                        # print(lambda_.lambda_.grad)
                                        optimizer_lambda._optimizer.set_mask(lambda_mask.clamp(max=1.0, min=0.0))
                                        optimizer_lambda.step()
                                        # print(step, lambda_().tolist())
                                        # input()
                                        lambda_.make_positive()
                                    
                                    

                                        # total_batchloss_for_lambda = total_loss_for_lambda.sum()
                                        # optimizer_lambda.backward(total_batchloss_for_lambda, retain_graph=True, scaler=scaler)
                                    
                                                                     
                                    # outputs.printparams()
                                    # input()
                                    
                                    
                                    # print(repeat_counts, allsat)
                                    if args.keywords in topics: 
                                        gen = primary_tokenizer.decode(pred_tokens[0].tolist())
                                        gen.replace("\n", " ")
                                        has_kw = False
                                        for tmp_kw in main_keywords: 
                                            if tmp_kw in gen:
                                                print(tmp_kw, gen)
                                                has_kw = True
                                        if has_kw:
                                            with open(f"mucola_kw_{args.keywords}.txt", "a") as f: 
                                                f.write(source_text + gen + "\n\n\n")
                                            predicted_allsat = True
                                            break

                                    cur_losses = []
                                    for b in range(batch_size):
                                        cur_loss = 0.0
                                        for beta, lossval in zip(betas, losses_for_backward):
                                            cur_loss = cur_loss + beta * lossval[b].item()     
                                        cur_losses.append(cur_loss)
                                        
                                        constrained = []
                                        allsat = True
                                        for i in range(1, len(losses)):
                                            if losses_for_backward[i] <= min_epsilons[i - 1]:
                                                constrained.append("sat")
                                            else:
                                                constrained.append("vio")
                                                allsat=False
                                            
                                        constrained = ",".join(constrained)

                                        modify_condition =\
                                            args.selection_criterion == "last" or\
                                            (best_loss[b] is None and args.selection_criterion == "weighted_sum") or\
                                            (best_loss[b] is not None and args.selection_criterion == "weighted_sum" and best_loss[b] > cur_loss)
                                        
                                        # print(repeat_counts, allsat, best_loss, best_allsat)
                                        if not modify_condition and args.selection_criterion == "mrr_allsat":
                                            modify_condition =\
                                                (best_loss[b] is None and allsat and repeat_counts[b] == 5) or\
                                                (best_loss[b] is not None and best_allsat[b] and allsat and repeat_counts[b] == 5)
                                            # print(modify_condition)
                                            # modify_condition = (best_loss[b] is not None and best_allsat[b] and allsat and repeat_counts[b] == 2)

                                        elif not modify_condition and args.selection_criterion == "primary_allsat":
                                            modify_condition =\
                                                (best_loss[b] is None and allsat) or\
                                                (best_loss[b] is not None and not best_allsat[b] and allsat) or\
                                                (best_allsat[b] and allsat and best_loss[b] > cur_loss)

                                        # step>20 and 

                                        if modify_condition:
                                            if args.dynamic_lr_update:
                                                print("resetting the learning rate and noise std, a constraint has been satisfied")
                                                cur_lr = optimizer._optimizer.update_lr(args.lr)
                                                # optimizer._optimizer.set_begin_std(0.01) #CHECK
                                            if args.selection_criterion != "last":
                                                print(f"modify condition @{step}", time.time()-starttime, end="\n")
                                            best_loss[b] = cur_loss
                                            best_allsat[b] = allsat
                                            best_repeat_count[b] = repeat_counts[b]
                                            for i in range(len(losses)):
                                                best_losses[i][b] = losses_for_backward[i][b].item()
                                            
                                            best_pred_tokens[b] = pred_tokens[b]
                                            best_index[b] = step
                                            # best_pred_probs[b] = (pred_probs[b].cpu(), logging_outputs[0]["lm_logprobs"][b])
                                            best_constrained = constrained
                                            best_step = step

                                            if args.show_all_outputs and len(losses) > 1 and allsat:
                                                best_prediction_set[b].add(target_sents[b])
                                        # elif best_step < step - 1 and args.dynamic_lr_update:
                                        #     print("resetting the learning rate, the constraint just got unsatisfied")
                                            
                                    if not args.time and step > 0 and step % args.log_interval == 0:
                                        if len(losses) > 1:
                                            log = f"beam cons: {predicted_allsat}; "
                                            log += f"Step {step}: lr:{cur_lr}; total_loss:{total_batchloss:.4f}; current [loss:{sum(cur_losses):.4f}; l:{','.join([f'{x:.4f}' for x in lambda_().tolist()])}; e:{','.join([f'{x:.4f}' for x in cur_epsilons])}; cons:{constrained}; "
                                            for i in range(len(losslists)):
                                                log = log + f" {lossabbr[i]}:{losslists[i][-1][-1]:.4f}; "
                                            
                                            if best_loss[0] is not None:
                                                log = log[:-1] + f"] |||| best [cur_loss:{sum(best_loss):.4f}; cons:{best_constrained};  "
                                                for i in range(len(best_losses)):
                                                    log = log + f"{lossabbr[i]}:{sum(best_losses[i]):.4f}; "
                                                log = log[:-1] + f"@ step #{best_index[-1]}" 
                                                log = log + "]"
                                            else:
                                                log = log[:-1] + f"] |||| best [none of the generations so far satisfies constraints]"
                                            print(log)
                                        else:
                                            log = f"Step {step}: lr:{cur_lr}; loss:{total_batchloss:.4f}; current [loss:{sum(cur_losses):.4f}; "
                                            for i in range(len(losslists)):
                                                log = log + f" {lossabbr[i]}:{losslists[i][-1][-1]:.4f}; "

                                            if best_loss[0] is not None:
                                                log = log[:-1] + f"] best [loss:{sum(best_loss):.4f} "
                                                for i in range(len(best_losses)):
                                                    log = log + f"{lossabbr[i]}:{sum(best_losses[i]):.4f}; "
                                                log = log[:-1] + f" at step {best_index[-1]}" 
                                                log = log + "]"
                                            else:
                                                log = log[:-1] + f"] |||| best [none of the generations so far satisfies constraints]"
                                            print(log, end="\n")

                                    del total_loss
                                    del total_batchloss
                                    for xx in losses_for_backward:
                                        del xx
                                    # del sats
                                    # del lambda_mask
                                    # if args.debug:
                                    #     print("LOSSFN SIZE",end=": ")
                                    #     for xx in lossfns:
                                    #         print(sys.getsizeof(xx), end=",")
                                    #     print()
                                    # torch.cuda.empty_cache()                                     

                                    if args.early_stop_steps > 0: #[0] is batch index, batch size in our case in 1 always so it doesn't matter.
                                        # print(args.selection_criterion)
                                        # print(lengthwise_best_prediction)
                                        early_stop_condition =\
                                            ("allsat" in args.selection_criterion and best_allsat[0]) or\
                                            (args.selection_criterion == "weighted_sum") or\
                                            (args.selection_criterion == "last")
                                        


                                        # print(early_stop_condition)
                                        if prev_loss is not None and abs(cur_loss - prev_loss) <= 1e-6:
                                            same_loss_count += 1
                                        else:   
                                            same_loss_count = 0

                                        if early_stop_condition and same_loss_count >= args.early_stop_steps:
                                            print(f"Early stop at @{step} with a loss value of {cur_loss} and satisfied constraints")
                                            break
                                        elif same_loss_count >= args.early_stop_steps + 100:#2 * args.lambda_update:
                                            print(f"Early stop at @{step} with a loss value of {cur_loss} and unsatisfied constraints")
                                            break
                                            
                                        prev_loss = cur_loss


                                except KeyboardInterrupt:
                                    print("skipping remaining optimizing steps and showing the best option so far")
                                    broken=True
                                    break

                            if args.time:
                                r = time.time()-starttime
                                print(r)
                                avg_time += r

                            predictions = []
                            prediction_idss = []
                            broken_skip = False
                            skip_printing = False
                            for b, item in enumerate(best_pred_tokens):
                                if item is None and broken:
                                    skip_printing = True
                                    if broken:
                                        broken_skip=input("Skip this input entirely? yes(y)/no(continue)/press ctrl+c to exit")
                                        broken_skip = broken_skip.lower() == "y"
                                        break
                                if (args.only_mucoco == "false" and not best_allsat[b]) or (item is None): #item is none happens when optimization fails
                                    prediction_ids = ", ".join([str(idx) for idx in AR_predicted_indices[0].tolist()])
                                    prediction_indices = AR_predicted_indices[0].tolist()
                                    prediction = AR_prediction

                                    lossvalue = 0.0
                                    for lossid in range(len(betas)):
                                        lossvalue += betas[lossid] * predictedlosslists[-1][lossid][b] # VERIFICATION NEEDED
                                    if lengthwise_best_prediction[b] is not None:
                                        print(f"{text_id} best prediction is from beam search, all constraints were not satisfied, allsat={lengthwise_best_prediction[b][2]}")
                                    else:
                                        print(f"{text_id} best prediction is empty since you set the option (only_mucoco=True) to never predict autoregressive outputs, all constraints were not satisfied, allsat=False")
                                else:
                                    prediction_ids = ", ".join([str(x) for x in target_prefix[b].tolist()])
                                    prediction_ids +=   f'[{", ".join([str(x) for x in item.tolist()])}]'
                                    prediction_indices = target_prefix[b].tolist() + item.tolist()
                                    
                                    targets = clean_output(item.tolist(), primary_tokenizer.eos_token_id, allow_first_eos=losses[0] == "bart")#, prompt=source_batch[b].unsqueeze(0), sentence_complete=True, lossfn=lossfns[0])
                                    if args.target_tokenize_different:
                                        with primary_tokenizer.as_target_tokenizer():
                                            prediction = primary_tokenizer.decode(target_prefix[b].tolist() + targets)
                                    else:
                                        prediction = primary_tokenizer.decode(target_prefix[b].tolist() + targets)

                                    print(f"{text_id} best prediction at step",best_index[b])
                                    lossvalue = best_loss[b]

                                    modify_condition =\
                                        lengthwise_best_prediction[b] is None or\
                                        (args.selection_criterion == "weighted_sum" and lengthwise_best_prediction[b][1] > lossvalue)
                                    
                                    if not modify_condition and args.selection_criterion == "primary_allsat":
                                        modify_condition =\
                                            (not lengthwise_best_prediction[b][2] and best_allsat[b]) or\
                                            (lengthwise_best_prediction[b][2] and best_allsat[b] and lengthwise_best_prediction[b][1] > lossvalue)
                                    
                                    elif not modify_condition and args.selection_criterion == "mrr_allsat":
                                        modify_condition =\
                                            (not lengthwise_best_prediction[b][2] and best_allsat[b] and best_repeat_count[b] >= 2) or\
                                            (lengthwise_best_prediction[b][2] and lengthwise_best_prediction[b][4] >= 2 and lengthwise_best_prediction[b][1] > lossvalue)
                                        
                                    
                                    if modify_condition:
                                        if args.debug:
                                            print("modify condition satisfied", end="\n")
                                        else:
                                            outallsatf.write("modify_condition satisfied ")
                                        lengthwise_best_prediction[b] = (prediction, lossvalue, best_allsat[b], prediction_indices, best_repeat_count[b])
                                prediction_idss.append(prediction_ids)
                                predictions.append(prediction)

                            if args.debug and not skip_printing:                    
                                for i, item in enumerate(best_pred_tokens):
                                    print(f"predicting length: {sent_length}")
                                    print("Given source:", source_text)
                                    print("Given target: ", target_text)
                                    print("Given additional: ", additional_text)
                                    print(f"Prediction ids: {prediction_ids}")
                                    print(f"Prediction: {prediction}; repeated {best_repeat_count[b]} times")
                                    print("All generations that satisfied the constraints: ", best_prediction_set[i])


                                    out = []
                                    # print(predictedlosslists)
                                    # input()
                                    # if target_batch is not None:
                                    #     for lossid in range(len(losses)):
                                    #         out.append(f"Gold {lossabbr[lossid]}: {predictedlosslists[lossid][-1]}")
                                    #out.append(f"Source {lossabbr[0]}: {source_primarylosslist[-1]}")
                                    # print("; ".join(out))

                                    out = []
                                    for lossid in range(len(losses)):
                                        out.append(f"{losses[lossid]}: {best_losses[lossid][i]}")
                                    print("; ".join(out))
                                
                                
                                if broken:
                                    broken_skip=input("Skip this input entirely? yes(y)/no(continue)/press ctrl+c to exit")
                                    broken_skip = broken_skip.lower() == "y"

                            all_stepcounts += best_index
                            # t = torch.cuda.get_device_properties(0).total_memory
                            # r = torch.cuda.memory_reserved(0)
                            # a = torch.cuda.memory_allocated(0)
                            # f = r-a  # free inside reserved

                            # print("###########MEMORY Before:", t, r, a, f)

                            # if args.debug:
                            #     t = torch.cuda.get_device_properties(0).total_memory
                            #     r = torch.cuda.memory_reserved(0)
                            #     a = torch.cuda.memory_allocated(0)
                            #     f = r-a  # free inside reserved
                            #     # print(f"###########MEMORY {step}:", t/(1024*1024*1024), r/(1024*1024*1024), a/(1024*1024*1024), f/(1024*1024*1024))
                            #     c = 0
                            #     d = 0
                            #     cur_garbage2 = Multiset()
                            #     for obj in gc.get_objects():
                            #         d+=1
                            #         try:
                            #             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                            #                 c+=1 #print(type(obj), obj.size())
                            #                 cur_garbage2.add(str(obj.dtype) + str(obj.size()))
                            #         except:
                            #             pass
                            #     print("before", c, d)
                            #     gc.collect()
                            #     if prev_garbage is not None:
                            #         diff_garbage = cur_garbage2.difference(prev_garbage)
                            #         cur_garbage.update(diff_garbage)
                            #         print(diff_garbage, len(diff_garbage))

                            #     else:
                            #         cur_garbage2 = Multiset()
                            #     import copy
                            #     prev_garbage = copy.deepcopy(cur_garbage2)
                            #     input(f"ok {len(prev_garbage)}")
                            #     # else:
                            #         #make first one empty again
                            #         # cur_garbage2 = Multiset()
                                
                                

                            optimizer.zero_grad(set_to_none=True)
                            del outputs
                            del optimizer
                            # del losses_for_backward
                            # del total_batchloss
                            if len(losses) > 1:
                                optimizer_lambda.zero_grad(set_to_none=True)
                                del optimizer_lambda
                                del lambda_
                            for modelname in loss2modelname.values():
                                if isinstance(name2model[modelname], tuple):
                                    name2model[modelname][0].zero_grad(set_to_none=True) 
                                    name2model[modelname][1].zero_grad(set_to_none=True) 
                                else:
                                    name2model[modelname].zero_grad(set_to_none=True)

                            # for loss in lossfns:
                            #     del loss
                            # lossfns = []
                            # for i, loss in enumerate(losses):
                            #     lossfns.append(lossbuilder.build_loss(loss, name2model[model_paths[i]], name2tokenizer[model_paths[i]], args))
                            #     loss2modelname[loss] = model_paths[i]
                            #     loss2tokenizer[loss] = name2tokenizer[model_paths[i]]

                            # torch.cuda.empty_cache()
                            # t = torch.cuda.get_device_properties(0).total_memory
                            # r = torch.cuda.memory_reserved(0)
                            # a = torch.cuda.memory_allocated(0)
                            # f = r-a  # free inside reserved
                            # print("###########MEMORY After:", t, r, a, f)


                            if args.debug and broken_skip: 
                                break
                            
                            if break_after:
                                break
                                        
                        
                        ### RESTART HERE
                        b=0
                        if lengthwise_best_prediction[b] is None or not lengthwise_best_prediction[b][2]: #constraints are not satisfied
                            if restart_idx < args.restarts: #atleast one more restart is left
                                continue #skip printing and loop over
                            elif lengthwise_best_prediction[b] is None:
                                lengthwise_best_prediction = [("", -1, False, [], -1)] #just blank which didn't satisfy the constraints
  
                        if args.debug:
                            if not skip_printing:
                                for b in range(batch_size):
                                    print("sample #"+str(sample_idx), f"repeat count: {lengthwise_best_prediction[b][4]}" , "best prediction for all lengths: ", lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n")
                        else:   
                            if args.output_style == "text":
                                for b in range(batch_size):
                                    outf.write(lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n")
                                    outf.flush()
                                    outallsatf.write(str(lengthwise_best_prediction[b][2]) + "\n")
                                    outallsatf.flush()
                            elif args.output_style == "jsonl":
                                for b in range(batch_size):
                                    if sample_idx == 0:
                                        output = {
                                            "prompt":{
                                                "text":source_text,
                                                "tokens":source_indices_write}, 
                                            "generations":[{
                                                "text": lengthwise_best_prediction[b][0],
                                                "tokens": lengthwise_best_prediction[b][3],
                                                "allsat": lengthwise_best_prediction[b][2],
                                                "repeat_count": lengthwise_best_prediction[b][4],
                                                "mucoco": True
                                                }]
                                        }
                                        if "commongen" in args.keywords or "roc" in args.keywords or "iate" in keywords:
                                            output['keywords'] = additional_text
                                    else:
                                        output['generations'].append(
                                            {
                                                "text": lengthwise_best_prediction[b][0],
                                                "tokens": lengthwise_best_prediction[b][3],
                                                "allsat": lengthwise_best_prediction[b][2],
                                                "repeat_count": lengthwise_best_prediction[b][4],
                                                "mucoco": True
                                            }
                                        )

                                if sample_idx + 1 == args.num_samples:
                                    json.dump(output, outf)
                                    outf.write("\n")
                                    outf.flush()

                                    outallsatf.write(str(lengthwise_best_prediction[b][2]) + "\n")
                                    outallsatf.flush()
                            elif args.output_style == "jsonl-summarize":
                                for b in range(batch_size): # for rouge
                                    if sample_idx == 0:
                                        output = {
                                            "article_id": str(text_id),
                                            "summarizer_id": 1,
                                            "summarize_type": "peer", 
                                            "article":{
                                                "text":source_text,
                                                "tokens":source_indices_write}, 
                                            "summary":{
                                                "text": [lengthwise_best_prediction[b][0]],
                                                "tokens": [lengthwise_best_prediction[b][3]],
                                                "allsat": [lengthwise_best_prediction[b][2]],
                                                "mucoco": [False]
                                                },
                                            "reference": {
                                                "text": target_text
                                            }
                                        }
                                        # print(output)
                                        
                                    else:
                                        output['summary']["text"].append(lengthwise_best_prediction[b][0])
                                        output['summary']["tokens"].append(lengthwise_best_prediction[b][3])
                                        output['summary']["allsat"].append(lengthwise_best_prediction[b][2])
                                        output['summary']["mucoco"].append(False)
                                    
                                    if args.keywords == "_cnndm_":
                                        if "keywords" in output['summary']:
                                            output['summary']['keywords'].append(additional_text)
                                        else:
                                            output['summary']['keywords'] = [additional_text]

                                print(f"SAMPLE IDX: {sample_idx}, NUM SAMPLES: {args.num_samples}")
                                if sample_idx + 1 == args.num_samples:
                                    json.dump(output, outf)
                                    outf.write("\n")
                                    outf.flush()
                                    #VERIFY
                        print(f"required output achieved or number of restarts ran out at attempt #{restart_idx+1}")
                        break # don't restart if already reached here

                    else: # skipping mucoco and writing autoregressive output 
                        if ask_skip != "y":
                            if args.debug:
                                print("Skipping this example. the autoregressive output already satisfies all the constraints or there's no constraints")
                                for b in range(batch_size):
                                    print("best prediction for all lengths: ", lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n")
                            else:
                                print(f"{text_id}: Skipping this example. the autoregressive output already satisfies all the constraints or there's no constraints")
                                if args.output_style == "text":
                                    for b in range(batch_size):
                                        outf.write(lengthwise_best_prediction[b][0].strip().replace("\n", " ") + "\n")
                                        outf.flush()
                                        outallsatf.write(str(lengthwise_best_prediction[b][2]) + "\n")
                                        outallsatf.flush()
                                elif args.output_style == "jsonl":
                                    for b in range(batch_size): #batch size is 1 so this is irrelevant
                                        if sample_idx == 0:
                                            output = {
                                                "prompt":{
                                                    "text":source_text,
                                                    "tokens":source_indices_write}, 
                                                "generations":[{
                                                    "text": lengthwise_best_prediction[b][0],
                                                    "tokens": lengthwise_best_prediction[b][3],
                                                    "allsat": lengthwise_best_prediction[b][2],
                                                    "mucoco": False
                                                    }]
                                            }
                                            # print(output)
                                        else:
                                            output['generations'].append(
                                                {
                                                    "text": lengthwise_best_prediction[b][0],
                                                    "tokens": lengthwise_best_prediction[b][3],
                                                    "allsat": lengthwise_best_prediction[b][2],
                                                    "mucoco": False
                                                }
                                            )
                                    if sample_idx + 1 == args.num_samples:
                                        json.dump(output, outf)
                                        outf.write("\n")
                                        outf.flush()
                                elif args.output_style == "jsonl-summarize":
                                    for b in range(batch_size): # for rouge
                                        if sample_idx == 0:
                                            output = {
                                                "article_id": str(text_id),
                                                "summarizer_id": 1,
                                                "summarize_type": "peer", 
                                                "article":{
                                                    "text":source_text,
                                                    "tokens":source_indices_write}, 
                                                "summary":{
                                                    "text": [lengthwise_best_prediction[b][0]],
                                                    "tokens": [lengthwise_best_prediction[b][3]],
                                                    "allsat": [lengthwise_best_prediction[b][2]],
                                                    "mucoco": [False]
                                                    },
                                                "reference": {
                                                    "text": target_text
                                                }
                                            }
                                            # print(output)
                                        else:
                                            output['summary']["text"].append(lengthwise_best_prediction[b][0])
                                            output['summary']["tokens"].append(lengthwise_best_prediction[b][3])
                                            output['summary']["allsat"].append(lengthwise_best_prediction[b][2])
                                            output['summary']["mucoco"].append(False)
                                    
                                    if sample_idx + 1 == args.num_samples:
                                        json.dump(output, outf)
                                        outf.write("\n")
                                        outf.flush()
                                        #VERIFY
                        break # don't restart
                
                    if args.debug and broken_skip:
                        break

                if args.debug and broken_skip: 
                    break

            del source_batch
            del target_batch
            del additional_batch
            del for_predicted_source_batch
            del predicted_batch
            source_batch = []
            target_batch = []
            for_predicted_source_batch = []
            additional_batch = []
            predicted_batch = []
            context_batch = []

    if args.outfile is not None:
        outf.close()
        outallsatf.close()
    print("average numbers of steps to converge =", np.mean(all_stepcounts))
    print("average time = ", avg_time/c)

def prune(text, length):
    # borrowed from COLD
    text = text.replace("\n", " ")
    import nltk
    sents = nltk.sent_tokenize(text)
    text_so_far = None
    length_so_far = 0
    for i, sent in enumerate(sents):
        text_so_far = sent if text_so_far is None else text_so_far + ' ' + sent
        sent_length = len(sent.split())
        length_so_far += sent_length
        if length_so_far >= length:
            break
    text_so_far_all.append(text_so_far)

def sentence_completion(prompt, tokens, lossfn):
    lossfn.args.max_output_length = lossfn.args.max_output_length + 10
    print(tokens)
    new_tokens = lossfn.generate(torch.cat([prompt, torch.LongTensor([tokens]).to(lossfn.device)], dim=1))
    print(new_tokens)
    lossfn.args.max_output_length = lossfn.args.max_output_length - 10
    return tokens + new_tokens[0].tolist()
    # return tokens

def clean_output(tokens, eos_token_id, return_tensors=False, allow_first_eos=False, remove_first=False, skip_special_tokens=[], prompt=None, sentence_complete=False, lossfn=None):
    # print(tokens)
    if sentence_complete:
        tokens = sentence_completion(prompt, tokens, lossfn)
    new_tokens = []
    if remove_first:
        tokens = tokens[1:]
    for i, tok in enumerate(tokens):
        if tok == eos_token_id and (not allow_first_eos or i > 0):
            break
        
        if (tok not in skip_special_tokens):
            new_tokens.append(tok)
        
    if return_tensors:
        return torch.LongTensor([new_tokens])
    return new_tokens
    
def cli_main():
    parser = options.get_parser()
    args = parser.parse_args()
    print(args)
    main(args)
