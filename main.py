from sentiment import sentiment_exp_loop
from keywords import keywords_loop
from detoxify import detoxify_loop
from eval import eval_loop
import argparse
import yaml


DEFAULT_PATHS = {
    'dlp': "configs/defaults/dlp.yaml",
    'bolt': "configs/defaults/bolt.yaml",
    'exp': "configs/exp_conf.yaml"
}

def conf_subparser(subparser, sampler): 
    default_conf = yaml.safe_load(open(DEFAULT_PATHS[sampler], 'r'))
    for key, val in default_conf.items():
        subparser.add_argument(f"--{key}", type=type(val), default=val)
    return subparser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='sampler')
    dlp_sampler = subparsers.add_parser('dlp')
    bolt_sampler = subparsers.add_parser('bolt')

    # general arguments 
    parser.add_argument("--prev_run_dir", default=None, type=str, required=False)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--exp", type=str, choices=['sentiment', 'detoxify', 'keywords'], required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--eval_on_fin", action='store_true')
    parser.add_argument("--device", type=str, default="cpu")

    conf_subparser(parser, 'exp')
    conf_subparser(dlp_sampler, 'dlp')
    conf_subparser(bolt_sampler, 'bolt')
    args = parser.parse_args()
    if args.prev_run_dir != None: 
        args.__dict__.update(yaml.safe_load(open(f"{args.prev_run_dir}/conf.yaml", 'r')))
    
    total_conf = args.__dict__
    if args.exp == "sentiment": 
        res = sentiment_exp_loop(total_conf)
    elif args.exp == "detoxify":
        res = detoxify_loop(total_conf)
    elif args.exp == "keywords":
        res = keywords_loop(total_conf)

    total_conf, generated_sentences = res 
    if args.eval_on_fin: 
        eval_loop(total_conf, generated_sentences)