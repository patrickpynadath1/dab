import faulthandler; faulthandler.enable()
import argparse
import yaml
import pickle
from evaluation import *


def exp_specific_metrics(exp, batch, **kwargs): 
    if exp == "sentiment": 
        return compute_sentiment(batch, **kwargs) 
    elif exp == "keywords":
        return [] 
    return  []

def eval_loop(total_conf, generated_sentences):

    cur_idx = 0
    batch_size = total_conf['batch_size']
    
    metrics = {
        'perp': [],
        'cola': [],
        'self_bleu': [],
        total_conf['exp']: []
    }
    cola_tokenizer, cola_model = load_cola_model()
    ext_tokenizer, ext_clf = load_external_sentiment()
    # cola_model.to(total_conf['device'])
    # ext_clf.to(total_conf['device'])
    while cur_idx < len(generated_sentences): 
        print(cur_idx)
        batch = generated_sentences[cur_idx:cur_idx+batch_size]
        
        metrics['perp'].append(compute_perplexity(batch))
        metrics['cola'].append(calc_cola(batch, cola_tokenizer, cola_model))
        metrics['self_bleu'].append(calc_self_bleu(batch))
        if total_conf['exp'] != "detoxify":
            metrics[total_conf['exp']].append(exp_specific_metrics(total_conf['exp'], batch, 
                                                        ext_tokenizer=ext_tokenizer, 
                                                        ext_clf=ext_clf))


        cur_idx += batch_size
    pickle.dump(metrics, open(f"{total_conf['prev_run_dir']}/eval_metrics_{total_conf['start_idx']}_{total_conf['end_idx']}.pkl", 'wb'))
    return