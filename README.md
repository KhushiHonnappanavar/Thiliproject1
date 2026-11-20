[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/gawahetkn3/badge.svg)](https://ask.thili.ai/b/gawahetkn3)
# Event Planning Request Sorter

Classifies a short client request into venue, catering, transport or equipment. Run commands from this folder using Python 3.11 and PowerShell.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python train.py
.\.venv\Scripts\python cli.py "Can we reserve the ballroom for Saturday?" --save results.jsonl
.\.venv\Scripts\python -m unittest -v
```

The first training run downloads pretrained DistilGPT2, a six-layer GPT-2 model. Trainer tunes its weights on 32 examples by predicting one of four custom label tokens after a classification marker. Eight separately worded requests are held out. Training saves the checkpoint and tokenizer in model/ and measured accuracy in runs/held_out_results.json. The small evaluation set suits a demonstration. Review predictions before assigning work. Inference reads the local checkpoint. Each call with --save appends one JSON record.
