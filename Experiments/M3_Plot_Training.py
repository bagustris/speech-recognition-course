import matplotlib.pyplot as plt
import os
import re
import argparse

# Module 3 lab: visualizing acoustic model training.
#
# This is the *scaffolding* for the lab. The complete, working version lives in
# Solutions/M3_Plot_Training.py -- use it to check your work after finishing.
#
# Your task (see M3_Acoustic_Modeling/README.md): implement plot_log_info() to
# read the training log written by M3_Train_AM.py and produce a two-panel plot:
#   top panel    : epoch vs. cross-entropy of the training set
#   bottom panel : epoch vs. frame error rate of the training AND dev sets
#
# The regular expressions that extract the epoch number, cross-entropy and
# frame-error metric from a log line are already provided below.


def plot_log_info(filename):

    re_ce = re.compile(r"loss = (?P<loss>[0-9]+\.[0-9]+)")
    re_ep = re.compile(r"Epoch\[(?P<ep>[0-9]+) of (?P<maxep>[0-9]+)")
    re_metric = re.compile(r"metric = (?P<metric>[0-9]+\.[0-9]+)")

    # Lists to accumulate per-epoch values.
    trainCE = []   # training cross-entropy
    trainFER = []  # training frame error rate
    cvFER = []     # development (cross-validation) frame error rate
    tr_ep = []     # epoch number for each training value
    cv_ep = []     # epoch number for each development value
    ep = 0
    with open(filename) as f:
        line = f.readline()
        while line:
            # TODO(M3): Parse each log line and populate the lists above.
            #
            # A training line starts with "Finished Epoch" and looks like:
            #   Finished Epoch[1 of 100]: [CE_Training] loss = 1.23 * 100,
            #       metric = 40.00% * 100 (5s);
            #   -> append `ep` to tr_ep, "loss" to trainCE, "metric" to trainFER
            #
            # A development line starts with "Finished Evaluation" and looks like:
            #   Finished Evaluation [20]: Minibatch[1-11573]: metric = 44.26% ...
            #   -> append the previous `ep` to cv_ep and "metric" to cvFER
            #
            # The regexes above are already compiled; usage examples:
            #   ep = int(re_ep.search(line).group("ep"))
            #   ce = float(re_ce.search(line).group("loss"))
            #   pe = float(re_metric.search(line).group("metric"))
            #
            # To tell the two line kinds apart, check whether the re.search of
            # "^Finished Epoch" or "^Finished Evaluation" matches. Remember to
            # advance the loop with: line = f.readline()
            line = f.readline()

    # TODO(M3): Draw the two-panel plot and save it to "fig/log.png".
    #
    # Create the figure with:
    #   fig, ax = plt.subplots(2, 1)
    # Top panel (ax[0]): tr_ep vs trainCE -- label x-axis "Epoch", y-axis
    # "Cross Entropy"; add a legend and grid.
    # Bottom panel (ax[1]): tr_ep vs trainFER (training) and cv_ep vs cvFER
    # (development) -- label y-axis "Frame Error Rate (%)"; add a legend/grid.
    # The fig directory may need to be created first (os.makedirs("fig",
    # exist_ok=True)), then save with plt.savefig("fig/log.png",
    # bbox_inches="tight").
    raise NotImplementedError(
        "TODO(M3): implement plot_log_info() in "
        "Experiments/M3_Plot_Training.py -- see the Module 3 lab instructions"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--log", help="training log file written by M3_Train_AM.py", required=True, default=None
    )
    args = parser.parse_args()
    plot_log_info(args.log)
