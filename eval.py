import faulthandler; faulthandler.enable()
import argparse
import yaml
import pickle
from evaluation import *


def exp_specific_metrics(exp, batch,  **kwargs): 
    if exp == "sentiment": 
        return compute_classifier_attribute(batch, **kwargs) 
    elif exp == "keywords":
        return []
    elif exp == "detoxify":
        return compute_classifier_attribute(batch, **kwargs) 
    return  []

def eval_loop(total_conf, generated_sentences, return_on_end=False, dump_on_end=True):

    cur_idx = 0
    # batch_size = total_conf['batch_size']
    batch_size = 200    
    metrics = {
        'perp': [],
        'cola': [],
        'self_bleu': [],
        total_conf['exp']: []
    }
    cola_tokenizer, cola_model = load_cola_model()
    ext_senti_tokenizer, ext_senti_clf = load_external_sentiment()
    ext_toxic_tokenizer, ext_toxic_clf = load_internal_toxic()
    cola_model.to(total_conf['device'])
    ext_senti_clf.to(total_conf['device'])
    ext_toxic_clf.to(total_conf['device'])
    metrics['perp'] = compute_perplexity(generated_sentences)
    while cur_idx < len(generated_sentences): 
        print(cur_idx)
        batch = generated_sentences[cur_idx:cur_idx+batch_size]
        
        # metrics['perp'].append(compute_perplexity(batch))
        # metrics['cola'].append(calc_cola(batch, cola_tokenizer, cola_model))
        # metrics['self_bleu'].append(calc_self_bleu(batch))
        if total_conf['exp'] != "detoxify":
            metrics[total_conf['exp']].append(exp_specific_metrics(total_conf['exp'], batch, 
                                                        ext_tokenizer=ext_senti_tokenizer, 
                                                        ext_clf=ext_senti_clf))
        else: 
            metrics[total_conf['exp']].append(exp_specific_metrics(total_conf['exp'], batch, 
                                                        ext_tokenizer=ext_toxic_tokenizer, 
                                                        ext_clf=ext_toxic_clf))


        cur_idx += batch_size
    if dump_on_end:
        pickle.dump(metrics, open(f"{total_conf['prev_run_dir']}/eval_metrics_{total_conf['start_idx']}_{total_conf['end_idx']}.pkl", 'wb'))
    if return_on_end:
        return metrics
    return