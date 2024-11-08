import soundfile as sf
import speech_sigproc as sp
import argparse
import htk_featio as htk
import os

data_dir = "./"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert audio files to speech recognition features in batch mode. "
        "Must specify train, dev, or test set. If train set is specified, "
        "global mean and variance of features are computed for use in acoustic model training.\n"
    )

    parser.add_argument(
        "-s",
        "--set",
        help="Specify which data to process, must be one of {train,dev,test}",
        required=True,
        default=None,
    )
    args = parser.parse_args()

    if args.set == "train":
        compute_stats = True
    else:
        compute_stats = False
    wav_list = os.path.join(data_dir, "lists/wav_{0}.list".format(args.set))
    feat_list = os.path.join(data_dir, "lists/feat_{0}.rscp".format(args.set))
    feat_dir = os.path.join(data_dir, "feat")
    # rscp_dir = "..."  # note ... is CNTK notation for "relative to the location of the list of feature files
    mean_file = os.path.join(data_dir, "am/feat_mean.ascii")
    invstddev_file = os.path.join(data_dir, "am/feat_invstddev.ascii")
    wav_dir = "../Experiments/"

    if not os.path.exists(os.path.join(data_dir, "am")):
        os.mkdir(os.path.join(data_dir, "am"))

    samp_rate = 16000
    fe = sp.FrontEnd(
        samp_rate=samp_rate, mean_norm_feat=True, compute_stats=compute_stats
    )
    # read lines

    with open(wav_list) as f:
        wav_files = f.readlines()
        wav_files = [x.strip() for x in wav_files]

    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)

    if not os.path.exists(os.path.dirname(feat_list)):
        os.makedirs(os.path.dirname(feat_list))
    out_list = open(feat_list, "w")
    count = 0
    for line in wav_files:

        wav_name = os.path.basename(line)
        root_name, wav_ext = os.path.splitext(wav_name)
        wav_file = os.path.join(wav_dir, line)
        feat_name = root_name + ".feat"
        feat_file = os.path.join(feat_dir, feat_name)
        x, s = sf.read(wav_file)

        if s != samp_rate:
            raise RuntimeError("Laboratory code assumes 16 kHz audio files!")

        feat = fe.process_utterance(x)
        htk.write_htk_user_feat(feat, feat_file)
        feat_rscp_line = os.path.join("feat", feat_name)
        print("Wrote", feat.shape[1], "frames to", feat_file)
        out_list.write("%s=%s[0,%d]\n" % (feat_name, feat_rscp_line, feat.shape[1] - 1))
        count += 1
    out_list.close()

    print("Processed", count, "files.")
    if compute_stats:
        m, p = fe.compute_stats()  # m=mean, p=precision (inverse standard deviation)
        htk.write_ascii_stats(m, mean_file)
        print("Wrote global mean to", mean_file)
        htk.write_ascii_stats(p, invstddev_file)
        print("Word global inv stddev to ", invstddev_file)
