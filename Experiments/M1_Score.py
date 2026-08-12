import argparse
import re
import wer

# Module 1 lab: write a speech recognition scoring program.
#
# This file is the *scaffolding* for the lab. The complete, working version
# lives in Solutions/M1_Score.py -- use it to check your work after finishing.
#
# Your task (see M1_Introduction/README.md -> "Lab for Module 1"): implement
# the score() function below to compute the Word Error Rate (WER) and Sentence
# Error Rate (SER) for a test corpus given a reference (correct) TRN file and
# a hypothesis (ASR output) TRN file.
#
# The provided wer module computes the edit distance between two strings of
# words:
#   tokens, errors, deletions, insertions, substitutions =
#       wer.string_edit_distance(ref=reference_string, hyp=hypothesis_string)


def parse_trn_line(line):
    """Parse a NIST TRN line 'word1 word2 ... (utt_id)' into (words, utt_id).

    The utterance id is the parenthesized token at the end of the line. If it is
    absent, utt_id is returned as None and the whole line is treated as words.
    """
    line = line.strip()
    match = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", line)
    if match:
        text, utt_id = match.group(1), match.group(2)
    else:
        text, utt_id = line, None
    return text.split(), utt_id


def read_trn(trn_file):
    """Read a TRN file into a list of (utt_id, words) tuples."""
    utterances = []
    with open(trn_file, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            words, utt_id = parse_trn_line(line)
            utterances.append((utt_id, words))
    return utterances


def score(ref_trn=None, hyp_trn=None):
    # TODO(M1): Complete this function to score a corpus of ASR output.
    #
    # 1. Read the reference and hypothesis transcriptions:
    #        ref = read_trn(ref_trn)
    #        hyp = read_trn(hyp_trn)
    #    each is a list of (utt_id, words) tuples.
    #
    # 2. Do NOT assume the two lists are in the same order. Match sentences by
    #    their utterance id (the parenthesized token). A convenient way is to
    #    index the hypothesis by id first:
    #        hyp_by_id = {utt_id: words for utt_id, words in hyp}
    #
    # 3. For every reference utterance, align it to its hypothesis words using
    #    the provided edit-distance function:
    #        tokens, errors, deletions, insertions, substitutions = \
    #            wer.string_edit_distance(ref=ref_words, hyp=hyp_words)
    #
    # 4. Aggregate the per-utterance counts across the whole corpus and report:
    #        - total number of reference sentences and how many had >= 1 error
    #          (Sentence Error Rate, SER = sentences_with_errors / N * 100)
    #        - total number of reference words (total_tokens)
    #        - total number of word errors (total_errors)
    #        - total substitutions, insertions and deletions
    #        - Word Error Rate (WER = total_errors / total_tokens * 100) and
    #          the percentage of substitutions / deletions / insertions
    #
    # The exact output format is up to you; a clear summary print statement is
    # sufficient. If a reference utterance id has no matching hypothesis, you
    # may raise an error or count it as all deletions -- your choice.
    raise NotImplementedError(
        "TODO(M1): implement score() in Experiments/M1_Score.py -- see the "
        "Module 1 lab instructions in M1_Introduction/README.md"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASR results.\n"
                                                 "Computes Word Error Rate and Sentence Error Rate")
    parser.add_argument('-ht', '--hyptrn', help='Hypothesized transcripts in TRN format', required=True, default=None)
    parser.add_argument('-rt', '--reftrn', help='Reference transcripts in TRN format', required=True, default=None)
    args = parser.parse_args()

    if args.reftrn is None or args.hyptrn is None:
        raise RuntimeError("Must specify reference trn and hypothesis trn files.")

    score(ref_trn=args.reftrn, hyp_trn=args.hyptrn)
