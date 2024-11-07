import struct
import numpy as np
import sys


def write_htk_user_feat(x, name="filename"):
    default_period = 100000  # assumes 0.010 ms frame shift
    num_dim = x.shape[0]
    num_frames = x.shape[1]
    hdr = struct.pack(
        ">iihh",        # the beginning '>' says write big-endian
        num_frames,     # nSamples
        default_period, # samplePeriod
        4 * num_dim,    # 2 floats per feature
        9,
    )  # user features

    out_file = open(name, "wb")
    out_file.write(hdr)

    for t in range(0, num_frames):
        frame = np.array(x[:, t], "f")
        if sys.byteorder == "little":
            frame.byteswap(True)
        frame.tofile(out_file)

    out_file.close()


def read_htk_user_feat(name="filename"):
    f = open(name, "rb")
    hdr = f.read(12)
    num_samples, samp_period, samp_size, parm_kind = struct.unpack(">IIHH", hdr)
    if parm_kind != 9:
        raise RuntimeError(
            "feature reading code only validated for USER feature type for this lab. There is other publicly available code for general purpose HTK feature file I/O\n"
        )

    num_dim = samp_size // 4

    feat = np.zeros([num_samples, num_dim], dtype=float)
    for t in range(num_samples):
        feat[t, :] = np.array(
            struct.unpack(">" + ("f" * num_dim), f.read(samp_size)), dtype=float
        )

    return feat


def write_ascii_stats(x, name="filename"):
    out_file = open(name, "w")
    for t in range(0, x.shape[0]):
        out_file.write("{0}\n".format(x[t]))
    out_file.close()
