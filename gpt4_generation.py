from openai import OpenAI
import argparse
import os

# make a text file with api key
# not the most secure way to do this, but it's fine for now
api_key = open("openai_api.txt", "r").readlines()[0]
client = OpenAI(api_key=api_key)
MODEL = "gpt-4o-mini"


def generate_samples(kw, topic, prompt, num_samples=10, model="gpt-4"):
    prompt = f"Given the topic {topic} and the keyword {kw}, write {num_samples} different, unique sentences using the keyword and relevant to the topic."
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def main(args):
    kw_topic_dct = {
        "computer": ["router", "Linux", "keyboard", "server"],
        "legal": ["plea", "subpoena", "transcript", "bankrupt"],
        "military": ["torpedo", "headquarters", "infantry", "battlefield"],
        "politics": ["court", "culture", "communism", "capitalism"],
        "religion": ["Bible", "church", "priest", "saint"],
        "science": ["microscope", "mass", "mineral", "scientist"],
        "space": ["meteor", "planet", "satellite", "astronaut"],
    }
    prompts = open("prompts/keywords_prompts_15.txt", "r").readlines()
    for i, p in enumerate(prompts):
        for topic, kws in kw_topic_dct.items():
            for kw in kws:
                print(f"Generating samples for {topic} and keyword {kw}")
                generation = generate_samples(
                    kw, topic, p, num_samples=args.num_samples
                )
                os.makedirs(f"keyword_ref_text/{topic}", exist_ok=True)
                with open(f"keyword_ref_text/{topic}/{kw}_{i}.txt", "w") as f:
                    f.write(generation)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples using GPT-4")
    parser.add_argument("--num_samples", type=int, default=30)
    args = parser.parse_args()

    main(args)
