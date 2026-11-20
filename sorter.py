import csv
from pathlib import Path

LABELS = ["venue inquiry", "catering request", "transport request", "equipment request"]
TOKENS = ["<|" + label.replace(" ", "_") + "|>" for label in LABELS]
MARKER = "<|classify|>"


def rows(split):
    with Path(__file__).with_name("data.csv").open(encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["split"] == split]


def encode(tokenizer, text):
    if not text.strip():
        raise ValueError("Enter a nonempty request.")
    ids = tokenizer.encode(text.strip(), add_special_tokens=False)[:95]
    return ids + [tokenizer.convert_tokens_to_ids(MARKER)]


def configure(tokenizer, model):
    tokenizer.add_special_tokens({"additional_special_tokens": [MARKER, *TOKENS]})
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
