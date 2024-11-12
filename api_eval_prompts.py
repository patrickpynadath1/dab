from eval import compute_perspective_scores, clean_for_eval


def main():
    detoxic_prompts = clean_for_eval(
        open(f"prompts/detoxic_prompts_1k.txt", "r").readlines()
    )
    print(f"num of sentences {len(detoxic_prompts)}")
    cur_batch = detoxic_prompts
    compute_perspective_scores(
        cur_batch,
        ".",
        start_idx=0,
        rate_limit=60,
    )


if __name__ == "__main__":
    main()
