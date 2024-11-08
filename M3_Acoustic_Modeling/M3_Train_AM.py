from __future__ import print_function
import numpy as np
import cntk as C
from cntk.train.training_session import *
from cntk.logging import *
import argparse

#model_type = "" # or "blstm"
data_dir = "../Experiments"
list_path = os.path.join(data_dir, "lists")
am_path = os.path.join(data_dir,"am")

globals = {
    "features_file": os.path.join(list_path,'feat_train.rscp'),
    "labels_file": os.path.join(am_path,'labels_all.cimlf'),
    "cv_features_file": os.path.join(list_path,'feat_dev.rscp'),
    "cv_labels_file": os.path.join(am_path,'labels_all.cimlf'),
    "label_mapping_file": os.path.join(am_path,'labels.ciphones'),
    "label_priors": os.path.join(am_path,'labels_ciprior.ascii'),
    "feature_mean_file": os.path.join(am_path,'feat_mean.ascii'),
    "feature_invstddev_file": os.path.join(am_path,'feat_invstddev.ascii'),
    "feature_dim": 40,
    "num_classes": 120
}


# frames to left/right of current frame to augment the input with
# for DNN total input size is 23 frames
# for RNN, just a single frames is used

# helper function to read ascii data files (for mean and inv stddev parameters)
def load_ascii_vector(file_name, var_name=None):
    x = np.asarray(np.loadtxt(file_name), dtype=np.float32)
    par = C.constant(value=x, shape=len(x), name=var_name)
    return par

def create_mb_source(features_file, labels_file, label_mapping_file, feature_dim, num_classes, max_sweeps=C.io.INFINITELY_REPEAT, context=(0,0), frame_mode=False):

    for file_name in [features_file, labels_file, label_mapping_file]:
        if not os.path.exists(file_name):
            raise RuntimeError(
                "File '{}' does not exist. Please check that datadir argument is set correctly.".format(file_name)
            )

    fd = C.io.HTKFeatureDeserializer(
        C.io.StreamDefs(features=C.io.StreamDef(shape=feature_dim, context=context, scp=features_file))
    )

    ld = C.io.HTKMLFDeserializer(
        label_mapping_file, C.io.StreamDefs(labels=C.io.StreamDef(shape=num_classes, mlf=labels_file))
    )

    return C.io.MinibatchSource([fd, ld], frame_mode=frame_mode, max_sweeps=max_sweeps)

def create_network(feature_dim = 40, num_classes=256, feature_mean_file=None, feature_inv_stddev_file=None,
                       feature_norm_files = None, label_prior_file = None, context=(0,0), model_type=None):

    def MyMeanVarNorm(feature_mean_file, feature_inv_stddev_file):
        m = C.reshape(load_ascii_vector(feature_mean_file,'feature_mean'), shape=(1, feature_dim))
        s = C.reshape(load_ascii_vector(feature_inv_stddev_file,'feature_invstddev'), shape=(1,feature_dim))
        def _func(operand):
            return C.reshape(C.element_times(C.reshape(operand,shape=(1+context[0]+context[1], feature_dim)) - m, s), shape=operand.shape)
        return _func


    def MyDNNLayer(hidden_size=128, num_layers=2):
        return C.layers.Sequential([
            C.layers.For(range(num_layers), lambda: C.layers.Dense(hidden_size, activation=C.sigmoid))
        ])

    def MyBLSTMLayer(hidden_size=128, num_layers=2):
        W = C.Parameter((C.InferredDimension, hidden_size), init=C.he_normal(1.0), name='rnn_parameters')
        def _func(operand):
            return C.optimized_rnnstack(operand, weights=W, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, recurrent_op='lstm' )
        return _func

    # Input variables denoting the features and label data
    feature_var = C.sequence.input_variable(feature_dim * (1+context[0]+context[1]))
    label_var = C.sequence.input_variable(num_classes)

    feature_norm = MyMeanVarNorm(feature_mean_file, feature_inv_stddev_file)(feature_var)
    label_prior = load_ascii_vector(label_prior_file, 'label_prior')
    log_prior = C.log(label_prior)

    if (model_type=="DNN"):
        net = MyDNNLayer(512,4)(feature_norm)
    elif (model_type=="BLSTM"):
        net = MyBLSTMLayer(512,2)(feature_norm)
    else:
        raise RuntimeError("model_type must be DNN or BLSTM")

    out = C.layers.Dense(num_classes, init=C.he_normal(scale=1/3))(net)

    # loss and metric
    ce = C.cross_entropy_with_softmax(out, label_var)
    pe = C.classification_error(out, label_var)
    ScaledLogLikelihood = C.minus(out, log_prior, name='ScaledLogLikelihood')

    # talk to the user
    C.logging.log_number_of_parameters(out)
    print()

    return {
        'feature': feature_var,
        'label': label_var,
        'output': out,
        'ScaledLogLikelihood': ScaledLogLikelihood,
        'ce': ce,
        'pe': pe,
        'final_hidden': net # adding last hidden layer output for future use in CTC tutorial
    }

def create_trainer(network, progress_writers, epoch_size):
    # Set learning parameters
    lr_per_sample = [1.0e-4] # transplanted schedule
    mm_time_constant = [2500] * 200
    lr_schedule = C.learning_rate_schedule(lr_per_sample, unit=C.learners.UnitType.sample, epoch_size=epoch_size)
    mm_schedule = C.learners.momentum_as_time_constant_schedule(mm_time_constant, epoch_size=epoch_size)

    momentum_sgd_learner = C.learners.momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule)

    # Create trainer
    return C.Trainer(network['output'], (network['ce'], network['pe']), [momentum_sgd_learner], progress_writers)


def train_and_test(network, trainer, train_source, minibatch_size, restore, model_path, model_name, epoch_size, cv_source):
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }
    cv_input_map = None if cv_source is None else {
        network['feature']: cv_source.streams.features,
        network['label']: cv_source.streams.labels
    }
    cv_checkpoint_interval = 5   # evaluate dev set after every N epochs

    checkpoint_config = CheckpointConfig(frequency=epoch_size,
                                         filename=os.path.join(model_path, model_name),
                                         restore=restore,
                                         preserve_all=True)

    cv_checkpoint_config = CrossValidationConfig(cv_source,
                                                 model_inputs_to_streams=cv_input_map,
                                                 frequency=epoch_size*cv_checkpoint_interval)

    # Train all minibatches
    training_session(
        trainer=trainer,
        mb_source=train_source,
        model_inputs_to_streams=input_map,
        mb_size=minibatch_size_schedule(minibatch_size, epoch_size),
        progress_frequency=epoch_size,
        checkpoint_config=checkpoint_config,
        cv_config=cv_checkpoint_config
    ).train()

def train_network(network, features_file, labels_file, label_mapping_file, max_epochs, minibatch_size=[256],
              restore=False, log_to_file=None, num_mbs_per_log=None, gen_heartbeat=False, profiling=False,
              cv_features_file = None, cv_labels_file = None, epoch_size=None, feature_dim=None, num_classes=None,
              model_path=None, context=(0,0), frame_mode=False, model_type=None):

    progress_writers = [C.logging.ProgressPrinter(freq=num_mbs_per_log,
                                                  tag='CE_Training',
                                                  log_to_file=log_to_file,
                                                  num_epochs=max_epochs
                                                  )]

    trainer = create_trainer(network, progress_writers, epoch_size)

    train_source = create_mb_source(features_file,
                                    labels_file,
                                    label_mapping_file,
                                    feature_dim=feature_dim,
                                    num_classes=num_classes,
                                    max_sweeps=max_epochs,
                                    context=context,
                                    frame_mode=frame_mode)

    cv_source = create_mb_source(cv_features_file,
                                 cv_labels_file,
                                 label_mapping_file,
                                 feature_dim=feature_dim,
                                 num_classes=num_classes,
                                 max_sweeps=1,
                                 context=context,
                                 frame_mode=frame_mode)

    train_and_test(
        network=network,
        trainer=trainer,
        train_source=train_source,
        minibatch_size=minibatch_size,
        restore=restore,
        model_path=model_path,
        model_name=model_type + "_CE",
        epoch_size=epoch_size,
        cv_source=cv_source
    )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--type', help='Network type to train (DNN or BLSTM)', required=True, default="DNN")
    args = parser.parse_args()

    model_type = str.upper(args.type)

    if model_type == "DNN":
        mb_size = [256]
        context_frames = (11,11)
        frame_mode = True
        max_epochs = 100
    elif model_type == "BLSTM":
        mb_size = [4096]
        context_frames = (0,0)
        frame_mode = False
        max_epochs = 1
    else:
        raise RuntimeError("type must be DNN or BLSTM")

    #mb_size = [256] if model_type == 'DNN' else [4096]
    #context_frames = (11, 11) if model_type == "DNN" else (0, 0)

    epoch_size = 1257284 # size of the corpus, 1 epoch = 1 pass thru the training data

    if not os.path.exists(am_path):
        raise RuntimeError("Cannot find am path, cannot proceed")

    model_path = os.path.join(am_path,model_type)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    log_file=os.path.join(model_path,"log")

    feature_dim=globals["feature_dim"]

    num_classes=globals["num_classes"]

    feature_mean_file = globals["feature_mean_file"]
    feature_inv_stddev_file= globals["feature_invstddev_file"]

    label_mapping_file = globals["label_mapping_file"]
    label_prior_file = globals["label_priors"]

    features_file = globals["features_file"]
    labels_file = globals["labels_file"]

    cv_features_file = globals["cv_features_file"]
    cv_labels_file = globals["cv_labels_file"]



    C.debugging.set_computation_network_trace_level(1)
    network = create_network(
            feature_dim=feature_dim, num_classes=num_classes,
            feature_mean_file=feature_mean_file, feature_inv_stddev_file=feature_inv_stddev_file,
            label_prior_file=label_prior_file,
            context=context_frames,
            model_type=model_type
            )

    train_network( network,
        features_file, labels_file, label_mapping_file,
        cv_features_file=cv_features_file, cv_labels_file=cv_labels_file,
        max_epochs=max_epochs,
        minibatch_size=mb_size,
        restore=True,
        log_to_file=log_file,
        num_mbs_per_log=1000,
        feature_dim=feature_dim,
        num_classes=num_classes,
        gen_heartbeat=False,
        epoch_size=epoch_size,
        model_path=model_path,
        context=context_frames,
        frame_mode=frame_mode,
        model_type=model_type
    )
    model = C.combine(network['ScaledLogLikelihood'],network['final_hidden'])
    model.save(os.path.join(model_path,model_type + "_CE_forCTC"))

if __name__ == '__main__':
    main()



