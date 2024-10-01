import argparse
import pandas as pd


def clean_text_outputs(raw_txt_file, prompt_file):
    lines = open(raw_txt_file, 'r').readlines()
    prompts = open(prompt_file, 'r').readlines()
    cleaned_lines = []
    for i, line in enumerate(lines):
        line = line.replace("\n", "")
        print(line)
        if line == "": 
            pass 
        else: 
            has_prompt = False 
            for prompt in prompts: 
                prompt = prompt.replace("\n", "")
                if prompt in line: 
                    has_prompt = True 
                    break
            if has_prompt:
                cleaned_lines.append(line)
            else: 
                cleaned_lines[-1] = cleaned_lines[-1] + line
    with open(f"cleaned_{raw_txt_file}", 'w') as f: 
        for line in cleaned_lines: 
            f.write(line)
            f.write("\n")


def clean_json_file(json_file, exp_name):
    json_lines = pd.read_json(json_file, lines=True).to_dict()
    cleaned_lines = []
    for i, prompt in json_lines['prompt'].items():
        print(prompt)
        for gen in json_lines['generations'][i]:
            gen_to_use = gen['text'].replace("\n", "")
            cur_line = prompt['text'] + gen_to_use
            cleaned_lines.append(cur_line)
    with open(f"cleaned_{exp_name}.txt", 'w') as f:
        for line in cleaned_lines:
            f.write(line)
            f.write("\n")
            
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean text outputs')
    parser.add_argument('--raw_txt_file', type=str, help='Path to raw text file')
    parser.add_argument('--prompt_file', type=str, help='Path to prompt file', default='prompts/sentiment_prompts_15.txt')
    parser.add_argument('--file_type', type=str, choices=['txt', 'json'], help='Type of file to clean')
    parser.add_argument('--exp_name', type=str, default='sentiment')
    args = parser.parse_args()
    if args.file_type == 'json':
        clean_json_file(args.raw_txt_file, args.exp_name)
    else: 
        clean_text_outputs(args.raw_txt_file, args.prompt_file)