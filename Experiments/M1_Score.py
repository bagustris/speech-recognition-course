import argparse
import re
import wer

# create a function that calls wer.string_edit_distance() on every utterance
# and accumulates the errors for the corpus. Then, report the word error rate (WER)
# and the sentence error rate (SER). The WER should include the the total errors as well as the
# separately reporting the percentage of insertions, deletions and substitutions.
# The function signature is
# num_tokens, num_errors, num_deletions, num_insertions, num_substitutions = wer.string_edit_distance(ref=reference_string, hyp=hypothesis_string)
#


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
    # read the reference and hypothesis transcriptions
    ref = read_trn(ref_trn)
    hyp = read_trn(hyp_trn)

    # initialize the total error counters
    total_tokens = 0
    total_errors = 0
    total_deletions = 0
    total_insertions = 0
    total_substitutions = 0
    sentence_errors = 0

    # if either file lacks utterance ids, fall back to positional alignment
    # (which requires equal line counts); otherwise match strictly by id
    positional = (any(utt_id is None for utt_id, _ in ref)
                  or any(utt_id is None for utt_id, _ in hyp))
    if positional:
        if len(ref) != len(hyp):
            raise RuntimeError(
                "Files without utterance ids must have the same number of lines "
                "(ref={}, hyp={}).".format(len(ref), len(hyp))
            )
        hyp_by_id = {}
    else:
        # index the hypotheses by utterance id so scoring is robust to ordering
        hyp_by_id = {utt_id: words for utt_id, words in hyp}
        if len(hyp_by_id) != len(hyp):
            raise RuntimeError("Duplicate utterance ids found in hypothesis file.")

    # loop over the reference transcriptions, matching hypotheses by id
    for i, (utt_id, ref_words) in enumerate(ref):
        if positional:
            hyp_words = hyp[i][1]
        elif utt_id in hyp_by_id:
            hyp_words = hyp_by_id[utt_id]
        else:
            raise RuntimeError(
                "Utterance id '{}' present in reference but missing from hypothesis.".format(utt_id)
            )

        # compute the errors for the current utterance
        tokens, errors, deletions, insertions, substitutions = wer.string_edit_distance(
            ref=ref_words, hyp=hyp_words
        )

        # print individual utterance scores
        print(f"id: ({utt_id})")
        print(f"Scores: N={tokens}, S={substitutions}, D={deletions}, I={insertions}\n")

        # accumulate the errors
        total_tokens += tokens
        total_errors += errors
        total_deletions += deletions
        total_insertions += insertions
        total_substitutions += substitutions
        if errors > 0:
            sentence_errors += 1

    # print summary statistics
    print("-----------------------------------")
    print("Sentence Error Rate:")
    print(f"Sum: N={len(ref)}, Err={sentence_errors}")
    print(f"Avg: N={len(ref)}, Err={sentence_errors/len(ref)*100:.2f}%")

    print("-----------------------------------")
    print("Word Error Rate:")
    print(f"Sum: N={total_tokens}, Err={total_errors}, Sub={total_substitutions}, Del={total_deletions}, Ins={total_insertions}")
    print(f"Avg: N={total_tokens}, Err={total_errors/total_tokens*100:.2f}%, Sub={total_substitutions/total_tokens*100:.2f}%, Del={total_deletions/total_tokens*100:.2f}%, Ins={total_insertions/total_tokens*100:.2f}%")
    print("-----------------------------------")
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluate ASR results.\n"
                                                 "Computes Word Error Rate and Sentence Error Rate")
    parser.add_argument('-ht', '--hyptrn', help='Hypothesized transcripts in TRN format', required=True, default=None)
    parser.add_argument('-rt', '--reftrn', help='Reference transcripts in TRN format', required=True, default=None)
    args = parser.parse_args()

    if args.reftrn is None or args.hyptrn is None:
        raise RuntimeError("Must specify reference trn and hypothesis trn files.")

    score(ref_trn=args.reftrn, hyp_trn=args.hyptrn)
