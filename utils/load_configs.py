import yaml
import re 


loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
    [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*([eE][-+][0-9]+)?
    |\\.[0-9_]+([eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*[eE][-+][0-9]+
    )$''', re.X),
    list(u'-+0123456789.')
)


def load_exp_conf(args):
    conf_path = f"{args.config_dir}/exp_conf.yaml"
    with open(conf_path, 'r') as stream:
        try:
            return yaml.load(stream, Loader=loader)
        except yaml.YAMLError as exc:
            print(exc)


def load_sampler_conf(args):
    conf_path = f"{args.config_dir}/{args.sampler}/{args.sampler_setup}.yaml"
    with open(conf_path, 'r') as stream:
        try:
            return yaml.load(stream, Loader=loader)
        except yaml.YAMLError as exc:
            print(exc)


# idea: have several default files for bolt, dlp, and experiment
# read for each experiment and each sampler, have a subparser or something to the cmdline 
# there should be four main cmdline args: sampler, experiment, and config file which points to the default 
# after this, the args are determined by cmdline

def setup_cmdline_args(default_path_dct):
    return 