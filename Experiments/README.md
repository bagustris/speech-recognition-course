# Experiments — Lab Scaffolding

This directory contains the **lab scaffolding** for the
[Speech Recognition Course](../README.md): working programs with the key pieces
left out for you to implement. Each unfinished script has a `TODO(M#)` block and
raises `NotImplementedError` until you complete it.

> [!TIP]
> Stuck, or want to check your work? The **complete** versions of every script
> live in [`../Solutions/`](../Solutions/README.md). Finish the lab first, then
> compare.

## How the labs work

1. Open the scaffolding script for the module you're on (see the table below).
2. Read the header comment and the `TODO(M#)` block — they describe exactly what
   to implement and which helper functions are already provided.
3. Fill in the missing code.
4. Run it (commands below). Compare your output against the reference in
   [`../Solutions/`](../Solutions/README.md).

Each module's `README.md` (`../M1_Introduction/`, `../M3_Acoustic_Modeling/`,
etc.) has the "Lab for Module N" write-up with the full background.

## Which scripts have exercises

| Script | Module | Your task |
| --- | --- | --- |
| `M1_Score.py` | M1 | Implement `score()` — compute WER and SER from a ref/hyp TRN pair |
| `M2_Wav2Feat_Single.py` | M2 | Complete single-file feature extraction |
| `M3_Train_AM.py` | M3 | Implement the training and evaluation loop in `train_model()` |
| `M3_Plot_Training.py` | M3 | Parse a training log and plot the curves |

The remaining files are provided complete as supporting code and are identical to
the versions in `../Solutions/`: `M2_Wav2Feat_Batch.py`, `arpa2fsa.py`,
`StaticDecoder.py`, and the helper modules `speech_sigproc.py`, `htk_featio.py`,
`wer.py`.

> [!NOTE]
> `am/`, `lists/`, and `misc/` here are **symlinks** into `../Solutions/`, so the
> data is shared and code that reads `am/`, `lists/`, or `misc/` relative to this
> directory works unchanged. A `git clone` on Linux/macOS reproduces the
> symlinks automatically; on Windows you may need Developer Mode / symlink
> support enabled.

## Prerequisites

Dependencies are managed with [uv](https://docs.astral.sh/uv/). From the
repository root:

```bash
uv sync
```

This installs `numpy`, `scipy`, `matplotlib`, `soundfile`, and a **CPU-only**
`torch` into `.venv`. Run scripts with `../.venv/bin/python` (from this
directory).

> [!IMPORTANT]
> M2 feature extraction and everything downstream need the **LibriSpeech**
> corpus, which is not bundled. The M1 lab uses the sample transcripts in
> `misc/` and needs no corpus.

## Running your work

Run all commands **from this `Experiments/` directory**.

```bash
# Module 1 — score the sample transcripts (no corpus needed)
../.venv/bin/python M1_Score.py -rt misc/ref.trn -ht misc/hyp.trn

# Module 2 — feature extraction (needs LibriSpeech)
../.venv/bin/python M2_Wav2Feat_Single.py

# Module 3 — train an acoustic model, capturing the log for plotting
../.venv/bin/python M3_Train_AM.py --type DNN | tee train_dnn.log
../.venv/bin/python M3_Plot_Training.py --log train_dnn.log
```

For the full pipeline (batch features → LM/FST → decoding → scoring) see the
step-by-step commands in [`../Solutions/README.md`](../Solutions/README.md).

## Checking your answer

After finishing a script, diff it against the reference implementation:

```bash
diff M1_Score.py ../Solutions/M1_Score.py
```

Differences in the header comments are expected; focus on the logic inside the
`TODO(M#)` blocks.
