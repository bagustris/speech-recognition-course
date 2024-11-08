import matplotlib.pyplot as plt
import re
import argparse


def plot_log_info(filename):

    re_ce = re.compile(r"loss = (?P<loss>[0-9]+\.[0-9]+)")
    re_ep = re.compile(r"Epoch\[(?P<ep>[0-9]+) of (?P<maxep>[0-9]+)")
    re_metric = re.compile(r"metric = (?P<metric>[0-9]+\.[0-9]+)")

    trainCE = []
    trainFER = []
    cvFER = []
    tr_ep = []
    cv_ep = []
    ep = 0
    with open(filename) as f:
        line = f.readline()
        while line:
            if re.search("^Finished Epoch", line) is not None:
                ep = int(re_ep.search(line).group("ep"))
                maxep = int(re_ep.search(line).group("maxep"))
                ce = float(re_ce.search(line).group("loss"))
                pe = float(re_metric.search(line).group("metric"))
                trainCE.append(ce)
                trainFER.append(pe)
                tr_ep.append(ep)
            elif re.search("^Finished Evaluation", line) is not None:
                pe = float(re_metric.search(line).group("metric"))
                cvFER.append(pe)
                cv_ep.append(ep)  # previous epoch
            # elif 'Finished Evaluation' in line:
            line = f.readline()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(tr_ep, trainCE, "b-", label="train")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Cross Entropy")
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(tr_ep, trainFER, "b-", label="train")
    ax[1].plot(cv_ep, cvFER, "r-", label="dev")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Frame Error Rate (%)")
    ax[1].legend()
    ax[1].grid(True)
    plt.savefig("fig/log.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--log", help="CNTK log file", required=True, default=None
    )
    args = parser.parse_args()
    plot_log_info(args.log)
