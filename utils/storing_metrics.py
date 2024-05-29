import os 
import yaml 


def initialize_metric_storing(total_conf, 
                              save_dir):
    # check if the directory exists
    counter = 1
    while os.path.exists(f"{save_dir}_{counter}"):
        # check if the directory has the same experimental setup
        # if it does, just save the metrics there
        # if it doesn't, create a new directory
        existing_conf = yaml.safe_load(open(f"{save_dir}_{counter}/conf.yaml", 'r'))
        if check_existing_setup(total_conf, existing_conf):
            return f"{save_dir}_{counter}" 
        counter += 1 
    os.makedirs(f"{save_dir}_{counter}")
    with open(f"{save_dir}_{counter}/conf.yaml", 'w') as f: 
        yaml.dump(total_conf, f)
    return f"{save_dir}_{counter}"


# if there is an existing directory with the same metrics, just save there instead 
# no need to have duplicate directories of the same experimental setup 
def check_existing_setup(old_conf, new_conf):
    for key in new_conf: 
        if key not in old_conf: 
            return False
        else: 
            if new_conf[key] != old_conf[key]: 
                return False
    return True


def initialize_best_loss(batch_size):
    minimum_loss = [100000] * batch_size
    stored_sentence = [""] * batch_size
    return minimum_loss, stored_sentence


def updating_best_loss(batch_size, 
                       loss,
                       sentences, 
                       min_loss_list, 
                       stored_sentence_list): 
    for i in range(batch_size):
        if loss[i] < min_loss_list[i]: 
            min_loss_list[i] = loss[i]
            stored_sentence_list[i] = sentences[i]
    return