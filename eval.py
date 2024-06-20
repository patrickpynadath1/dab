import faulthandler; faulthandler.enable()
import argparse
import yaml
import pickle
from evaluation import *


def exp_specific_metrics(exp, batch, **kwargs): 
    if exp == "sentiment": 
        return compute_sentiment(batch, **kwargs) 
    elif exp == "toxicity": 
        return compute_toxicity_score(batch)
    elif exp == "keywords":
        return [] 
    return  []

def main(args):
    total_conf = yaml.safe_load(open(f"{args.run_dir}/conf.yaml", 'r'))
    generated_sentences = open(f"{args.run_dir}/output.txt", 'r').readlines()

    cur_idx = 0
    batch_size = total_conf['batch_size']
    
    metrics = {
        'perp': [],
        'cola': [],
        'self_bleu': [],
        args.exp: []
    }
    cola_tokenizer, cola_model = load_cola_model()
    ext_tokenizer, ext_clf = load_external_sentiment()
    while cur_idx < len(generated_sentences): 
        print(cur_idx)
        batch = generated_sentences[cur_idx:cur_idx+batch_size]
        
        # metrics['perp'].append(compute_perplexity(batch))
        metrics['cola'].append(calc_cola(batch, cola_tokenizer, cola_model))
        metrics['self_bleu'].append(calc_self_bleu(batch))
        metrics[args.exp].append(exp_specific_metrics(args.exp, batch, 
                                                      ext_tokenizer=ext_tokenizer, 
                                                      ext_clf=ext_clf))


        cur_idx += batch_size
    pickle.dump(metrics, open(f"{args.run_dir}/eval_metrics.pkl", 'wb'))
    return 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    # the run dir where all the generated sentences are stored
    parser.add_argument("--run_num", type=int, required=True)
    parser.add_argument("--exp", type=str, choices=['sentiment', 'toxicity', 'keywords'], required=True)
    parser.add_argument("--sampler", type=str, choices=['bolt', 'dlp'], default='bolt')
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    args.run_dir = f"{args.results_dir}/{args.exp}/{args.sampler}_{args.run_num}"
    # print('asd')
    main(args)
