# Experiment log

## 2026-07-19 — Milestone 1 environment audit

| Item | Observed value |
| --- | --- |
| Operating system | Windows 11, 64-bit |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU VRAM | 4096 MiB |
| NVIDIA driver | 576.40 |
| CUDA compute capability | 8.6 |
| Logical CPU cores | 20 |
| Free space on C: | 153.7 GiB |
| System RAM | Not available from non-elevated audit |
| Default Python | CPython 3.13.1 at `C:\Python313\python.exe` |
| PyTorch | 2.13.0+cpu in project `.venv`; not installed globally |
| Test framework | Standard-library `unittest`; no external test dependency |

### Interpretation

The 4 GB GPU is suitable for the bigram, character-Transformer, and small pilot
milestones. Later models will require conservative batch sizes, gradient
accumulation, mixed precision, and measured memory use. The 10–30M parameter
target is plausible, but its final configuration must be chosen after a pilot
run rather than assumed now.

### Milestone 1 verification

The initial implementation passed all seven tests: four tokenizer tests, two
bigram tests, and an end-to-end training/checkpoint/generation test. The
tiny-batch test drove the bigram loss below `0.02`, verifying that gradients,
loss, and optimizer updates work together.

An end-to-end CPU smoke run trained for 300 steps on the bundled sample:

| Step | Training loss | Validation loss |
| ---: | ---: | ---: |
| 0 | 4.2520 | 4.1560 |
| 100 | 2.1318 | 2.8008 |
| 200 | 2.1069 | 2.8473 |
| 300 | 2.0831 | 2.9303 |

The checkpoint saved and reloaded successfully, and generation produced
character-level fragments. The rising validation loss after step 100 is useful
evidence of overfitting on this deliberately tiny corpus, not a reason to tune
this baseline extensively.

CUDA is not yet enabled because the default Python package source supplied a
CPU-only PyTorch wheel. GPU package selection will be handled as a separate,
verified environment step before the character-Transformer pilot.
