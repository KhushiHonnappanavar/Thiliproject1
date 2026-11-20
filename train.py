import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, set_seed
from sorter import LABELS, TOKENS, configure, encode, rows


def main():
    set_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    base = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = GPT2LMHeadModel.from_pretrained(base)
    configure(tokenizer, model)
    token_ids = tokenizer.convert_tokens_to_ids(TOKENS)
    # Start each new label embedding near its category word in GPT-2's vocabulary.
    with torch.no_grad():
        weights = model.get_input_embeddings().weight
        for label, token_id in zip(LABELS, token_ids):
            words = tokenizer.encode(" " + label.split()[0])
            weights[token_id].copy_(weights[words].mean(0))

    def examples(split):
        return [{"input_ids": encode(tokenizer, row["text"]),
                 "labels": LABELS.index(row["label"])} for row in rows(split)]

    def collate(batch):
        padded = tokenizer.pad([{"input_ids": r["input_ids"]} for r in batch],
                               return_tensors="pt")
        padded["labels"] = torch.tensor([r["labels"] for r in batch])
        return padded

    class LabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"]
            mask = inputs["attention_mask"]
            logits = model(input_ids=inputs["input_ids"], attention_mask=mask).logits
            scores = logits[torch.arange(len(labels), device=labels.device), mask.sum(1)-1][:, token_ids]
            loss = torch.nn.functional.cross_entropy(scores, labels)
            return (loss, {"logits": scores}) if return_outputs else loss

    args = TrainingArguments(output_dir="runs", num_train_epochs=12,
                             learning_rate=5e-5, per_device_train_batch_size=4,
                             save_strategy="no", report_to=[], seed=42)
    trainer = LabelTrainer(model=model, args=args, train_dataset=examples("train"),
                           eval_dataset=examples("test"), data_collator=collate,
                           compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(1) == p.label_ids).mean().item()})
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_metrics("held_out", metrics)
    trainer.save_model("model")
    tokenizer.save_pretrained("model")
    print(metrics)


if __name__ == "__main__":
    main()
