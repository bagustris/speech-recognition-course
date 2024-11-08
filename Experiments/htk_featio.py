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


def HTKFeatureConfiguration(stream_name, scp_file, dimension, left_context=0, 
                            right_context=0, broadcast=False,
                            defines_mb_size=False, max_sequence_length=65535):
    """
    Creates an HTK feature configuration object.
    
    Args:
        stream_name: name of the feature stream
        scp_file: path to the SCP file containing HTK feature file paths
        dimension: dimension of the feature vectors
        left_context: number of frames of left context to include
        right_context: number of frames of right context to include
        broadcast: whether to broadcast the features
        defines_mb_size: whether this stream defines minibatch size
        max_sequence_length: maximum allowed sequence length
    """
    config = {
        'stream_name': stream_name,
        'scp_file': scp_file,
        'dimension': dimension,
        'left_context': left_context,
        'right_context': right_context,
        'broadcast': broadcast,
        'defines_mb_size': defines_mb_size,
        'max_sequence_length': max_sequence_length
    }
    return config

def htk_feature_deserializer(feature_configs):
    """
    Creates an HTK feature deserializer from feature configurations.
    
    Args:
        feature_configs: list of HTK feature configuration objects
    """
    deserializer = {
        'feature_configs': feature_configs,
        'type': 'htk'
    }
    return deserializer

def HTKFeatureDeserializer(streams):
    '''
    Configures the HTK feature reader that reads speech data from scp files.

    Args:
        streams: any dictionary-like object that contains a mapping from stream
          names to :class:`StreamDef` objects. Each StreamDef object configures
          a feature stream.
    '''
    feat = []
    for stream_name, stream in streams.items():
        if stream.stream_alias is not None:
            raise ValueError("HTKFeatureDeserializer does not support stream names")
        if 'scp' not in stream:
            raise ValueError("No scp files specified for HTKFeatureDeserializer")
        dimension = stream.dim
        scp_file = stream['scp']
        broadcast = stream['broadcast'] if 'broadcast' in stream else False
        defines_mb_size = stream.get('defines_mb_size', False)
        max_sequence_length = stream.get('max_sequence_length', 65535)
        left_context, right_context = stream.context if 'context' in stream\
                                                     else (0, 0)
        htk_config = HTKFeatureConfiguration(stream_name, scp_file,
                                                     dimension, left_context,
                                                     right_context, broadcast,
                                                     defines_mb_size, max_sequence_length)
        feat.append(htk_config)

    if len(feat) == 0:
        raise ValueError("no feature streams found")
    return htk_feature_deserializer(feat)
