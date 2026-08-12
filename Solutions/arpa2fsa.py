import re
import codecs
import gzip
import numpy as np
import argparse

def main():

    parser = argparse.ArgumentParser(description='Transform gzipped ARPA file into an approximate FST.')

    parser.add_argument('arpa_file', help='the source language model')
    parser.add_argument('out_file_base', help='we will write output to here, adding .tfsa and .sym extensions')
    parser.add_argument('--prune_5k', '-prune_5k', help='artificially limit to the top 5000 unigrams in the model', action='store_true')
    args = parser.parse_args()

    lm = arpalm(args.arpa_file)
    lm.make_fst(args.prune_5k)
    lm.write_fst_body(args.out_file_base + '.tfsa')
    lm.write_fst_symbols(args.out_file_base + '.sym')

    pass

class arpalm:
    def __init__(self, filename):
        re_data = re.compile(r'^\\data\\')
        re_ngram_count = re.compile(r'^ngram (\d+)=(\d+)')
        re_ngrams = re.compile(r'^\\(\d+)-grams:')
        re_end = re.compile(r'^\\end\\')

        ngram_counts = []
        current_ngram = 0
        ngrams = {}

        def found_data_section(line:str):
            match_object = re_data.search(line)
            if match_object is not None:
                return True
            return False
        def found_ngrams_section(line:str):
            nonlocal current_ngram
            match_object = re_ngrams.search(line)
            if match_object is not None:
                current_ngram = int(match_object.group(1))
                return True
            return False
        def found_ngram_counts(line:str):
            nonlocal ngram_counts
            match_object = re_ngram_count.search(line)
            if match_object is not None:
                ngram_size = int(match_object.group(1))
                ngram_counts.append(int(match_object.group(2)))
                assert (len(ngram_counts) == ngram_size)
        def found_end_section(line: str):
            match_object = re_end.search(line)
            if match_object is not None:
                return True
            return False
        def record_ngram(line: str):
            nonlocal ngram_counts
            nonlocal current_ngram
            nonlocal ngrams

            parts = line.lower().split()
            if len(parts) == 0:
                return

            # count down from the number of expected ngrams
            ngram_counts[current_ngram - 1] -= 1

            ngram_score = float(parts[0])
            ngram = parts[1:current_ngram + 1]
            backoff_score = float(parts[current_ngram + 1] if len(parts) > current_ngram + 1 else 0)

            ngrams[tuple(ngram)] = (ngram_score, backoff_score)

        found_end = False
        with codecs.getreader('UTF-8')(open(filename, 'rb')) as f:

            # ignore the header, looking for start of data
            for line in f:
                if found_data_section(line):
                    break

            # parse the header
            for line in f:
                if found_ngrams_section(line):
                    break
                elif found_ngram_counts(line):
                    pass
            assert (current_ngram == 1)

            # parse the ngram data
            for line in f:
                # handle start of new section
                if found_ngrams_section(line):
                    continue
                # are we done?
                if found_end_section(line):
                    found_end = True
                    break
                record_ngram(line)

        # sanity checks. did we find hte end? did we read the expected number of ngrams?
        assert (found_end)
        for i in ngram_counts:
            assert(i == 0)

        self.max_ngram = len(ngram_counts)
        self.ngrams = ngrams

    def make_fst(self, prune_5k=False):

        arcs = [(0,1,'<s>','<s>',0)]
        state_map = {
            ('unreachable_start_state',): 0,
            ('<s>',): 1,
            (): 2
        }
        state_count = 3
   
        if prune_5k:
            if len([x for x in self.ngrams if len(x)==1]) <= 5000:
                prune_5k = False
            else:
                unigram_thresh = sorted([self.ngrams[x][0] for x in self.ngrams if len(x)==1], reverse=True)
                unigram_thresh = unigram_thresh[5000]
                unigrams = [x[0] for x in self.ngrams if len(x)==1 and self.ngrams[x][0]>unigram_thresh]

        def ngram_to_state(ngram: tuple):
            nonlocal state_count
            nonlocal state_map
            if ngram not in state_map:
                state_map[ngram] = state_count
                state_count += 1
            return state_map[ngram]

        for ngram in self.ngrams:
            if prune_5k and ngram[-1] not in unigrams:
                continue
            word = ngram[-1]
            hist = ngram[:-1]
            futr = ngram[1-self.max_ngram:]
            score = self.ngrams[ngram]
            assert(len(hist) == 0 or hist in self.ngrams)
            # insert the ngram arc
            src = ngram_to_state(hist)
            dst = ngram_to_state(futr)
            label = word
            if score[0] > -99:
                arcs.append((src, dst, word, label, -np.log10(np.exp(1)) * score[0]))

            # insert the backoff arc
            if len(ngram) < self.max_ngram and word != '</s>':
                src = ngram_to_state(ngram)
                dst = ngram_to_state(ngram[1:])
                word = '<gamma>'
                label = word
                if score[1] > -99:
                    arcs.append((src, dst, word, label, -np.log10(np.exp(1)) * score[1]))

        # look for missing backoff arcs
        for ngram in filter(lambda x: len(x) > 1, state_map):
            pretend_ngram = ngram[1-self.max_ngram:]
            if pretend_ngram not in self.ngrams:
                assert(pretend_ngram[1:] in self.ngrams)  # make sure we don't need recursion
                src = ngram_to_state(pretend_ngram)
                dst = ngram_to_state(pretend_ngram[1:])
                word = '<gamma>'
                label = word
                arcs.append((src, dst, word, label, 0))

        # find and record all end states
        end_states = {}
        for arc in arcs:
            if arc[2] == '</s>':
                end_states[arc[1]] = 1

        self._end_states = end_states
        self._fst = arcs

    def write_fst_body(self, fst_body_filename):
        with open(fst_body_filename, 'wt') as f:
            f.write('\n'.join(['{:d} {:d} {} {} {:f}'.format(int(x[0]), int(x[1]), x[2], x[3], float(x[4])) for x in self._fst]))
            f.write('\n')
            f.write('\n'.join(['{:d}'.format(int(x)) for x in self._end_states]))

    def write_fst_symbols(self, fst_symbol_filename):
        syms = set()
        num_symbols = 1

        def process_sym(sym):
            nonlocal num_symbols
            nonlocal syms
            if sym not in syms:
                syms.add(sym)
                f.write('{} {}\n'.format(sym, num_symbols))
                num_symbols += 1

        with open(fst_symbol_filename, 'wt') as f:
            f.write('<eps> 0\n')
            for arc in self._fst:
                process_sym(arc[2])
                process_sym(arc[3])

    def score_ngram(self, ngram):
        ngram = tuple(map(lambda x:x[1:], ngram))
        score = 0
        while(len(ngram)):
            ng = ngram[1-3:]
            while len(ng):
                if ng in self.ngrams:
                    score = score + self.ngrams[ng][0]
                    ng=()
                else:
                    score = score + self.ngrams[ng[:-1]][1]
                    ng = ng[1:]
            ngram = ngram[:-1]
        return score

if __name__ == '__main__':
    main()