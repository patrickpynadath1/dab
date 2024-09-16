import faulthandler

faulthandler.enable()
import argparse
import yaml
import pickle
from evaluation import *
from models import load_sentiment_discriminator, load_tokenizer

def exp_specific_metrics(exp, batch, **kwargs):
    if exp == "sentiment":
        return compute_classifier_attribute(batch, **kwargs)
    elif exp == "keywords":
        return []
    elif exp == "detoxify":
        return compute_classifier_attribute(batch, **kwargs)
    return []


def eval_loop(total_conf, generated_sentences, return_on_end=False, dump_on_end=True):
    torch.cuda.set_device(total_conf["device"])
    cur_idx = 0
    # batch_size = total_conf['batch_size']
    batch_size = 300
    metrics = {"perp": [], "cola": [], "self_bleu": [], total_conf["exp"]: [], 'internal_senti': [], 'sst_senti': []}
    cola_tokenizer, cola_model = load_cola_model()
    ext_senti_tokenizer, ext_senti_clf = load_external_sentiment()
    ext_toxic_tokenizer, ext_toxic_clf = load_internal_toxic()
    internal_sentiment_tokenizer = load_tokenizer()
    internal_sentiment_clf = load_sentiment_discriminator()
    sst_tok, sst_clf = load_external_sentiment("textattack/roberta-base-SST-2")
    print(sst_tok)
    print(sst_clf)
    print(internal_sentiment_clf)
    print(internal_sentiment_tokenizer)
    internal_sentiment_tokenizer.pad_token = internal_sentiment_tokenizer.eos_token
    # internal_sentiment_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # sst_tok.add_special_tokens({'pad_token': '[PAD]'})
    cola_model.to(total_conf["device"])
    ext_senti_clf.to(total_conf["device"])
    ext_toxic_clf.to(total_conf["device"])
    internal_sentiment_clf.to(total_conf["device"])
    sst_clf.to(total_conf["device"])
    metrics["perp"] = compute_perplexity(generated_sentences)
    while cur_idx < len(generated_sentences):
        print(cur_idx)
        batch = generated_sentences[cur_idx : cur_idx + batch_size]

        metrics['cola'].append(compute_classifier_attribute(
                                            batch,
                                            ext_tokenizer=cola_tokenizer,
                                            ext_clf=cola_model)
                                )
        if total_conf["exp"] != "detoxify":
            metrics[total_conf["exp"]].append(
                exp_specific_metrics(
                    total_conf["exp"],
                    batch,
                    ext_tokenizer=ext_senti_tokenizer,
                    ext_clf=ext_senti_clf,
                )
            )
            if total_conf["exp"] == "sentiment":

                metrics['internal_senti'].append(
                    exp_specific_metrics(
                        total_conf["exp"],
                        batch,
                        ext_tokenizer=internal_sentiment_tokenizer,
                        ext_clf=internal_sentiment_clf,
                    )            
                )
                metrics['sst_senti'].append(
                    exp_specific_metrics(
                        total_conf["exp"],
                        batch,
                        ext_tokenizer=sst_tok,
                        ext_clf=sst_clf,
                    )            
                )
        else:
            metrics[total_conf["exp"]].append(
                exp_specific_metrics(
                    total_conf["exp"],
                    batch,
                    ext_tokenizer=ext_toxic_tokenizer,
                    ext_clf=ext_toxic_clf,
                )
            )
            
        

        cur_idx += batch_size
    if dump_on_end:
        pickle.dump(
            metrics,
            open(
                f"{total_conf['prev_run_dir']}/eval_metrics_{total_conf['start_idx']}_{total_conf['end_idx']}.pkl",
                "wb",
            ),
        )
    if return_on_end:
        return metrics
    return
