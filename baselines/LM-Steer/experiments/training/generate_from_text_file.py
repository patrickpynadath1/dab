import json
from tqdm import tqdm
import torch

from lm_steer.arguments import parse_args
from lm_steer.models.get_model import get_model


def generate(prompt_data, steer_values, tokenizer, model,
             prompt_num, prompt_length, num_beams, num_beam_groups,
             do_sample, temperature, top_p, device):
    total_generations = []
    for prompt_text in tqdm(prompt_data):
        prompt_text = prompt_text.replace("\n", "")
        token_length = tokenizer(prompt_text,
                                 return_tensors="pt")["input_ids"].shape[1]
        for _i in range(prompt_num):
            output = model.generate(
                prompt_text,
                steer_values,
                seed=_i,
                max_length=token_length+prompt_length,
                min_length=token_length+prompt_length,
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            output = output[len(prompt_text):]
            total_generations.append(prompt_text + output.replace("\n", ""))
        # if args.verbose:
        #     print(prompt_text)
    return total_generations


def main(args):
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    model, tokenizer = get_model(
        args.model_name, args.adapted_component, args.adaptor_class,
        args.num_steers,
        args.rank, args.epsilon, args.init_var, args.low_resource_mode)
    model.to_device(device)

    ckpt = torch.load(args.ckpt_name)
    model.load_state_dict(ckpt[1])

    # predicting sentences
    prompt_data =open(args.eval_file, "r").readlines() 

    model.eval()
    prompt_num = 20
    prompt_length = 20
    if args.eval_size is not None:
        prompt_data = prompt_data[:args.eval_size]
    num_beams = 1
    num_beam_groups = 1
    do_sample = True
    temperature = args.temperature
    steer_values = list(map(float, args.steer_values)) \
        if args.steer_values is not None else None

    total_generations = generate(prompt_data, steer_values, tokenizer, model, prompt_num,
             prompt_length, num_beams, num_beam_groups, do_sample, temperature,
             args.top_p, device)

    with open(args.output_file, "w") as f:
        for _prompt in total_generations:
            f.write(_prompt + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
