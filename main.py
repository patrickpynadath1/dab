from sentiment import sentiment_exp_loop
from keywords import keywords_loop
from detoxify import detoxify_loop
from abductive_reasoning import abductive_reasoning_loop
from eval import eval_loop, compute_perspective_scores, clean_for_eval
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
    eval_only = subparsers.add_parser('eval_only')
    api_eval = subparsers.add_parser('api_eval')
    experiments = ['sentiment', 'detoxify', 'keywords', 'abductive_reasoning']

    # general arguments 
    parser.add_argument("--prev_run_dir", default=None, type=str, required=False)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--exp", type=str, choices=experiments, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--eval_on_fin", action='store_true')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf_file", type=str, default=None)
    api_eval.add_argument("--rate_limit", type=int, default=60)
    conf_subparser(parser, 'exp')
    conf_subparser(dlp_sampler, 'dlp')
    conf_subparser(bolt_sampler, 'bolt')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    args = parser.parse_args()
    initial_mode = args.sampler
    initial_prev_run_dir = args.prev_run_dir
    if args.prev_run_dir != None: 
        args.__dict__.update(yaml.safe_load(open(f"{args.prev_run_dir}/conf.yaml", 'r')))
    if args.conf_file != None:
        args.__dict__.update(yaml.safe_load(open(args.conf_file, 'r')))
    total_conf = args.__dict__
    if initial_mode == "bolt" or initial_mode == "dlp": 
        if args.exp == "sentiment": 
            res = sentiment_exp_loop(total_conf)
        elif args.exp == "detoxify":
            res = detoxify_loop(total_conf)
        elif args.exp == "keywords":
            res = keywords_loop(total_conf)
        elif args.exp == 'abductive_reasoning': 
            res = abductive_reasoning_loop(total_conf)

        total_conf, generated_sentences = res 
        if args.eval_on_fin: 
            eval_loop(total_conf, generated_sentences)
    elif initial_mode == "eval_only":
        gen_sentences = clean_for_eval(open(f"{initial_prev_run_dir}/output.txt", "r").readlines())
        print(f"num of sentences {len(gen_sentences)}")
        cur_batch = gen_sentences[args.start_idx:args.end_idx]
        total_conf['prev_run_dir'] = initial_prev_run_dir
        print(f"eval gen sentences {total_conf['start_idx']} to {total_conf['end_idx']}")
        eval_loop(total_conf, cur_batch)
    elif initial_mode == "api_eval": 
        gen_sentences = clean_for_eval(open(f"{initial_prev_run_dir}/output.txt", "r").readlines())
        print(f"num of sentences {len(gen_sentences)}")
        cur_batch = gen_sentences[args.start_idx:args.end_idx]
        if args.exp == 'detoxify': 
            compute_perspective_scores(cur_batch, initial_prev_run_dir, start_idx=args.start_idx, rate_limit=args.rate_limit)