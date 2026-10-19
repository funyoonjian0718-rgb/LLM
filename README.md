# TinyIssueLM

TinyIssueLM is a learning project that builds a language model from randomly
initialized weights. It does not load or fine-tune a pretrained model.

Milestone 1 deliberately uses the smallest useful pipeline:

- a deterministic Unicode character tokenizer;
- document-level train/validation splitting before tokenizer training;
- a neural bigram language model implemented with PyTorch;
- next-token cross-entropy training;
- atomic checkpoint saving and local text generation;
- unit tests, including a mandatory tiny-batch overfitting test.

## Why begin with a bigram model?

For a token `t`, the bigram model learns one row of scores describing which
token is likely to follow `t`. Softmax converts those scores to probabilities.
Cross-entropy penalizes low probability on the true next token, and
backpropagation changes the scores. A Transformer uses much richer context, but
the tokenizer, shifted targets, loss, optimizer, checkpoints, and generation
loop introduced here all carry forward.

## Environment

Create an isolated environment and install the project:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Installing PyTorch downloads packages but does not require an API key or incur
an API charge. No cloud GPU is used by this repository.

## Verify the milestone

```powershell
python -m unittest discover -s tests -v
```

The important check is `test_can_overfit_one_tiny_batch`. Its final loss must
fall below `0.02`; otherwise the learning loop is not ready to scale.

## Train and generate

The sample corpus consists of original project text and is safe to redistribute.
Blank-line-separated blocks are treated as documents. Validation documents are
held out before the tokenizer is trained.

```powershell
tiny-issue-lm train --text data/sample.txt --output artifacts/bigram.pt `
  --steps 1000 --context-length 32

tiny-issue-lm generate --checkpoint artifacts/bigram.pt `
  --prompt "A small" --max-new-tokens 200 --top-k 10
```

The bigram model sees only one preceding character, so its output will be
locally plausible but not coherent. That limitation is expected and provides a
baseline for the character-level Transformer milestone.

The first local audit installed a CPU-only PyTorch wheel. The code selects CUDA
automatically when a CUDA-enabled PyTorch build is present, but GPU enablement
is intentionally deferred until the Transformer pilot needs it.

## Data rules carried into later milestones

- Split by document before training the tokenizer.
- Train the tokenizer only on the training documents.
- Append an explicit end-of-text token to each document.
- Map unseen validation characters to an explicit unknown token.
- Keep seeds and preprocessing deterministic.
- Do not commit large or restricted datasets.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the running environment and experiment
log.
