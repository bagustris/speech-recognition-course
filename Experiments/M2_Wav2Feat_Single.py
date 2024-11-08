import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import htk_featio as htk
import speech_sigproc as sp

data_dir = "."
wav_file = "./LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
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

    # plot waveform
    plt.plot(x)
    plt.title("waveform")
    plt.savefig("fig/waveform.png", bbox_inches="tight")
    plt.close()

    # plot mel filterbank
    for i in range(0, fe.num_mel):
        plt.plot(fe.mel_filterbank[i, :])
    plt.title("mel filterbank")
    plt.savefig("fig/mel_filterbank.png", bbox_inches="tight")
    plt.close()

    # plot log mel spectrum (fbank)
    plt.imshow(
        feat, origin="lower", aspect=4
    )  # flip the image so that vertical frequency axis goes from low to high
    plt.title("log mel filterbank features (fbank)")
    plt.savefig("fig/fbank.png", bbox_inches="tight")
    plt.close()

htk.write_htk_user_feat(feat, feat_file)
print("Wrote {0} frames to {1}".format(feat.shape[1], feat_file))

# if you want to verify, that the file was written correctly:
feat2 = htk.read_htk_user_feat(name=feat_file)
print("Read {0} frames rom {1}".format(feat2.shape[1], feat_file))
print("Per-element absolute error is {0}".format(np.linalg.norm(feat.T-feat2)/(feat2.shape[0]*feat2.shape[1])))
