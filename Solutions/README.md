# Solutions — Complete Experiments

This directory contains the **complete, runnable reference implementations** for the
labs in the [Speech Recognition Course](../README.md). Every script here works
end to end; use them to run the full ASR pipeline, or to check your work after
finishing the scaffolding in [`../Experiments/`](../Experiments/).

> [!NOTE]
> Looking for the *lab exercises*? The scaffolding code with `TODO`s to complete
> lives in [`../Experiments/`](../Experiments/README.md). This directory holds
> the finished answers.

## Contents

| Script | Module | What it does |
| --- | --- | --- |
| `M1_Score.py` | M1 | Scores ASR output — Word Error Rate (WER) and Sentence Error Rate (SER) |
| `M2_Wav2Feat_Single.py` | M2 | Extracts features (MFCC/filterbank) from a single wav file |
| `M2_Wav2Feat_Batch.py` | M2 | Batch feature extraction for the train/dev/test sets |
| `M3_Train_AM.py` | M3 | Trains a DNN or BLSTM acoustic model in PyTorch |
| `M3_Plot_Training.py` | M3 | Plots loss/frame-error curves from a training log |
| `arpa2fsa.py` | M4 | Converts an ARPA language model to an FST |
| `StaticDecoder.py` | M5 | Viterbi WFST decoder that produces hypotheses |

Shared helper modules (imported by the scripts above): `speech_sigproc.py`
(feature front end), `htk_featio.py` (HTK feature I/O), `wer.py` (edit distance).

The `am/`, `lists/`, and `misc/` directories hold the data these scripts read.
(In `Experiments/` these same directories are symlinks pointing back here.)

## Prerequisites

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and declared in
[`../pyproject.toml`](../pyproject.toml). From the repository root:

```bash
uv sync
```

This installs `numpy`, `scipy`, `matplotlib`, `soundfile`, and a **CPU-only**
build of `torch` into `.venv`. Run scripts with `../.venv/bin/python` (from this
directory) or `uv run python`.

> [!IMPORTANT]
> Feature extraction (M2) and everything downstream of it need the
> **LibriSpeech** corpus, which is not included in this repository. The
> file lists in `lists/` point at the expected wav locations. Steps that only
> use the bundled sample data (M1 scoring, M4 LM conversion) run without it.

## Running the experiments

Run all commands **from this `Solutions/` directory**.

### Module 1 — Scoring (WER/SER)

Uses the bundled sample transcripts in `misc/`; no corpus needed.

```bash
../.venv/bin/python M1_Score.py -rt misc/ref.trn -ht misc/hyp.trn
```

The output should match `misc/expected_result.txt` (WER 44.44% on the sample).

### Module 2 — Feature extraction

Single file — note this script's paths (`wav_file`, output `feat/`) are
hard-coded relative to `../data/`, so edit the `wav_file`/`data_dir`
variables at the top to point at a wav you actually have. 
You can also softlink your LibriSpeech corpus into `../data/LibriSpeech` to match the expected path.  

```bash
../.venv/bin/python M2_Wav2Feat_Single.py
```

Batch mode — process a whole set. Running the `train` set also computes the
global feature mean/variance used for acoustic-model training:

```bash
../.venv/bin/python M2_Wav2Feat_Batch.py --set train
../.venv/bin/python M2_Wav2Feat_Batch.py --set dev
../.venv/bin/python M2_Wav2Feat_Batch.py --set test
```

### Module 3 — Acoustic model training

Train a DNN (default) or a BLSTM, capturing the log so it can be plotted:

```bash
../.venv/bin/python M3_Train_AM.py --type DNN  | tee train_dnn.log
../.venv/bin/python M3_Train_AM.py --type BLSTM | tee train_blstm.log
```

Plot the training/validation curves from a captured log:

```bash
../.venv/bin/python M3_Plot_Training.py --log train_dnn.log
```

### Module 4 — Language model to FST

```bash
../.venv/bin/python arpa2fsa.py path/to/lm.arpa.gz decoding_graph
```

Writes `decoding_graph.tfsa` (the FST) and `decoding_graph.sym` (symbol table).
Add `--prune_5k` to prune to a 5k vocabulary.

### Module 5 — Decoding

Combine the trained acoustic model with the decoding graph to produce
hypotheses, then score them with the Module 1 tool:

`M3_Train_AM.py` saves the checkpoint to `am/<TYPE>/<TYPE>_CE.pt` (e.g.
`am/DNN/DNN_CE.pt`), so point `-am` at that file:

```bash
../.venv/bin/python StaticDecoder.py \
    -am am/DNN/DNN_CE.pt \
    -decoding_graph decoding_graph.tfsa \
    -label_map am/labels.ciphones \
    -scp lists/feat_test.rscp \
    -trn hyp.trn \
    -lmweight 10 -beam_width 5000

../.venv/bin/python M1_Score.py -rt misc/ref.trn -ht hyp.trn
```

## Legacy CNTK script

`M3_Train_AM_cntk.py` is the original CNTK version kept for reference. CNTK is
unmaintained and not installable on Python 3.12 — use the PyTorch
`M3_Train_AM.py` above instead.
