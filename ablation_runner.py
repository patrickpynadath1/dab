from multiprocessing import Process, Queue, Pool, Manager
import yaml 
from detoxify import detoxify_loop
from keywords import keywords_loop
from sentiment import sentiment_exp_loop
from eval import eval_loop
import argparse
from copy import deepcopy
from utils import initialize_metric_storing
import time 
import pickle 
import torch 


def load_default_conf(sampler, conf_path='configs/defaults'): 
    with open(f"{conf_path}/{sampler}.yaml", 'r') as f: 
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return conf


def load_exp_conf(exp_conf_path='configs/exp_conf.yaml'): 
    with open(exp_conf_path, 'r') as f: 
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
    return exp_conf


def load_ablation_conf(ablation_conf_path): 
    with open(ablation_conf_path, 'r') as f: 
        ablation_conf = yaml.load(f, Loader=yaml.FullLoader)
    return ablation_conf

# should also include making all the directories in this function 
def construct_all_configs(ablation_conf, 
                          base_conf, 
                          sampler, 
                          experiments, 
                          base_dir='results'):
    configs_to_run = []
    # configuring default setting
    for k, v in ablation_conf['default'].items():
        base_conf[k] = v
    # configuring params to ablate over 
    for exp in experiments:
        for ablation_k, ablation_v in ablation_conf.items():
            if ablation_k  == 'default':
                continue
            for ablation_val in ablation_v['values']:
                conf = deepcopy(base_conf)
                conf['sampler'] = sampler
                conf['exp'] = exp
                conf[ablation_k] = ablation_val
                if exp == 'keywords':
                    conf['num_steps'] = 50
                else: 
                    conf['num_steps'] = 8
                if exp == 'sentiment':
                    save_dir = initialize_metric_storing(conf, f"{base_dir}/{conf['exp']}_pos/{sampler}")
                else: 
                    save_dir = initialize_metric_storing(conf, f"{base_dir}/{conf['exp']}/{sampler}")
                conf['prev_run_dir'] = save_dir
                configs_to_run.append((save_dir, conf))
    return configs_to_run


def worker_writing_data(data_queue):
    (save_dir, data) = data_queue.get()
    with open(save_dir, "wb") as f: 
        pickle.dump(data, f)
    print(f"dumped data for {save_dir}")
    return


def worker_run_exp(run_dir, total_conf, gpu_queue, to_save_queue): 
    # try: 
    print(f"running job for {run_dir}")
    gpu_id = gpu_queue.get()
    device = f"cuda:{gpu_id}"
    total_conf['device'] = device
    try: 
        if total_conf['exp'] == 'detoxify': 
            res = detoxify_loop(total_conf)
        elif total_conf['exp'] == 'keywords': 
            res = keywords_loop(total_conf, dump_sampling_metrics=False, return_sampling_metrics=True)
        elif total_conf['exp'] == 'sentiment': 
            res = sentiment_exp_loop(total_conf, dump_sampling_metrics=False, return_sampling_metrics=True)
        total_conf, generated_sentences, sampling_metrics = res
        to_save_queue.put((f"{run_dir}/sampling_metrics.pkl", sampling_metrics))
        total_conf['prev_run_dir'] = run_dir
        metrics = eval_loop(total_conf, generated_sentences, return_on_end=True, dump_on_end=False)
        to_save_queue.put((f"{run_dir}/eval_metrics_abl.pkl", metrics))
        print("sent to save queue")
    except Exception as e:
        print(f"error in {run_dir}")
        print(e)
    with torch.cuda.device(device): 
        torch.cuda.empty_cache()
    gpu_queue.put(gpu_id)
    print("done with job")
    return
    


def run_ablations(all_conf, avail_gpus, jobs_per_gpu): 
    m = Manager()
    gpu_queue = m.Queue()
    jobs_queue = m.Queue()
    to_save_queue = m.Queue()
    for gpu_id in avail_gpus: 
        for _ in range(jobs_per_gpu):
            gpu_queue.put(int(gpu_id))
    for conf in all_conf:
        jobs_queue.put(conf)
    # run the experiments
    pool = Pool(len(avail_gpus)*jobs_per_gpu+1)
    while True:
        # the only break condition -- we only want to break 
        # once all the metrics are stored and there are no more jobs
        if to_save_queue.empty() and jobs_queue.empty():
            break
        if not to_save_queue.empty(): 
            pool.apply_async(worker_writing_data, args=(to_save_queue,))

        # if there are gpus and jobs, run the job
        if not jobs_queue.empty() and not gpu_queue.empty():
            (run_dir, total_conf) = jobs_queue.get()
            pool.apply_async(worker_run_exp, args=(run_dir, total_conf, gpu_queue, to_save_queue),)
            time.sleep(.1)
        if not jobs_queue.empty() and gpu_queue.empty():
            time.sleep(5)
    pool.close()
    # res = pool.map(run_exp, [(c[0], c[1], gpu_queue) for c in all_conf])
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ablation experiments')
    parser.add_argument('--ablation_conf', type=str, required=True)
    parser.add_argument('--exp', type=str, nargs='+', required=True)
    parser.add_argument('--gpu_ids', type=int, nargs='+', required=True)
    parser.add_argument('--sampler', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=False, default='results')
    parser.add_argument('--jobs_per_gpu', type=int, default=1)
    args = parser.parse_args()

    ablation_conf = load_ablation_conf(args.ablation_conf)
    exp_conf = load_exp_conf()
    base_conf = load_default_conf(args.sampler)
    base_conf = {**base_conf, **exp_conf}
    base_conf['exp'] = args.exp
    base_conf['save_dir'] = args.save_dir
    all_configs = construct_all_configs(ablation_conf, base_conf, args.sampler, experiments=args.exp)
    run_ablations(all_configs, args.gpu_ids, args.jobs_per_gpu)
