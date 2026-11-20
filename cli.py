import argparse
import json
from pathlib import Path

from sorter import LABELS, TOKENS, encode


def save_result(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Classify an event planning request")
    parser.add_argument("request")
    parser.add_argument("--save", type=Path, help="Append the classification as JSON Lines")
    args = parser.parse_args()
    if not args.request.strip():
        parser.error("Enter a nonempty request.")
    if not Path("model/config.json").is_file():
        parser.error("Train the classifier first with python train.py")
    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel
    tokenizer = AutoTokenizer.from_pretrained("model", local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained("model", local_files_only=True).eval()
    with torch.inference_mode():
        logits = model(torch.tensor([encode(tokenizer, args.request)])).logits[0, -1]
        scores = logits[tokenizer.convert_tokens_to_ids(TOKENS)]
    record = {"request": args.request, "label": LABELS[scores.argmax().item()]}
    if args.save:
        save_result(args.save, record)
    print(record["label"])


if __name__ == "__main__":
    main()
