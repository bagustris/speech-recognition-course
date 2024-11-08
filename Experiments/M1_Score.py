import argparse
import wer

# create a function that calls wer.string_edit_distance() on every utterance
# and accumulates the errors for the corpus. Then, report the word error rate (WER)
# and the sentence error rate (SER). The WER should include the the total errors as well as the
# separately reporting the percentage of insertions, deletions and substitutions.
# The function signature is
# num_tokens, num_errors, num_deletions, num_insertions, num_substitutions = wer.string_edit_distance(ref=reference_string, hyp=hypothesis_string)
#

def score(ref_trn=None, hyp_trn=None):
    # read the reference and hypothesis transcriptions
    ref = open(ref_trn, 'r').readlines()
    hyp = open(hyp_trn, 'r').readlines()
    
    # initialize the total error counters
    total_tokens = 0
    total_errors = 0
    total_deletions = 0
    total_insertions = 0
    total_substitutions = 0
    sentence_errors = 0
    
    # loop over the reference and hypothesis transcriptions
    for i in range(len(ref)):
        # compute the errors for the current utterance
        tokens, errors, deletions, insertions, substitutions = wer.string_edit_distance(ref=ref[i], hyp=hyp[i])
        
        # print individual utterance scores
        print(f"id: ({str(i).zfill(4)}-{str(i).zfill(6)}-{str(i).zfill(4)})")
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
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluate ASR results.\n"
                                                 "Computes Word Error Rate and Sentence Error Rate")
    parser.add_argument('-ht', '--hyptrn', help='Hypothesized transcripts in TRN format', required=True, default=None)
    parser.add_argument('-rt', '--reftrn', help='Reference transcripts in TRN format', required=True, default=None)
    args = parser.parse_args()

    if args.reftrn is None or args.hyptrn is None:
        RuntimeError("Must specify reference trn and hypothesis trn files.")

    score(ref_trn=args.reftrn, hyp_trn=args.hyptrn)

