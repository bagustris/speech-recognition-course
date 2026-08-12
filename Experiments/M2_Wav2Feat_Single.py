import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import htk_featio as htk
import speech_sigproc as sp

# Module 2 lab: feature extraction for speech recognition.
#
# This is the *scaffolding* for the lab. The complete, working version lives in
# Solutions/M2_Wav2Feat_Single.py -- use it to check your work after finishing.
#
# Your task (see M2_Speech_Signal_Processing/README.md -> "Lab"): first fill in
# the missing methods of the FrontEnd class in speech_sigproc.py, then complete
# the plotting section below so that running this script produces three figures:
#   fig/waveform.png        (the raw audio waveform)
#   fig/mel_filterbank.png  (one line per mel filterbank band)
#   fig/fbank.png           (the log mel filterbank features)
#
# Everything before the plotting section (reading the audio file, running the
# FrontEnd feature extractor, and writing/verifying the HTK feature file) is
# provided for you.

data_dir = "../Experiments"
wav_file = "../Experiments/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
feat_file = os.path.join(data_dir, "feat/1272-128104-0000.feat")
plot_output = True

if not os.path.isfile(wav_file):
    raise RuntimeError(
        "input wav file is missing. Have you downloaded the LibriSpeech corpus?"
    )

if not os.path.exists(os.path.join(data_dir, "feat")):
    os.mkdir(os.path.join(data_dir, "feat"))

samp_rate = 16000

x, s = sf.read(wav_file)
if s != samp_rate:
    raise RuntimeError("LibriSpeech files are 16000 Hz, found {0}".format(s))

fe = sp.FrontEnd(samp_rate=samp_rate, mean_norm_feat=True)


feat = fe.process_utterance(x)

if plot_output:
    if not os.path.exists("fig"):
        os.mkdir("fig")

    # TODO(M2): Plot and save the three requested figures.
    #
    # 1. Waveform: plot the raw audio samples and save to "fig/waveform.png".
    #        plt.plot(x)
    #        plt.title("waveform")
    #        plt.savefig("fig/waveform.png", bbox_inches="tight")
    #        plt.close()
    #
    # 2. Mel filterbank: draw one line per mel band using the FrontEnd's
    #    precomputed filterbank weights (fe.mel_filterbank[i, :]) and save to
    #    "fig/mel_filterbank.png".
    #
    # 3. Log mel filterbank features ("fbank"): display the feature matrix
    #    produced above and save to "fig/fbank.png". Use
    #        plt.imshow(feat, origin="lower", aspect=4)
    #    (origin="lower" flips the image so the vertical frequency axis goes
    #    from low to high).
    #
    # Remember to call plt.close() after each figure so they do not accumulate.
    raise NotImplementedError(
        "TODO(M2): implement the feature plots in "
        "Experiments/M2_Wav2Feat_Single.py -- see the Module 2 lab instructions"
    )

htk.write_htk_user_feat(feat, feat_file)
print("Wrote {0} frames to {1}".format(feat.shape[1], feat_file))

# if you want to verify, that the file was written correctly:
feat2 = htk.read_htk_user_feat(name=feat_file)
print("Read {0} frames from {1}".format(feat2.shape[0], feat_file))
print("Per-element absolute error is {0}".format(np.linalg.norm(feat.T-feat2)/(feat2.shape[0]*feat2.shape[1])))
