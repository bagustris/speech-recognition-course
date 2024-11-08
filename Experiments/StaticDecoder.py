import sys
sys.path.append("../M2_Speech_Signal_Processing")
import os.path
import cntk
import numpy as np
import scipy.sparse
import scipy.misc
import re
import time
import itertools
from collections import namedtuple
import argparse
from htk_featio import read_htk_user_feat
from typing import List
import operator

def feature_stacker(x, context_frames=11):
    """This function processes isolated frame-based feature vectors into windowed stacks of neighboring frames.

    This is useful for deep neural network acoustic models, which are often trained with an input window of
    adjacent acoustic data instead of isolated frames.

    Args:
        x (array): Frame-based acoustic data, one frame per row.
        context_frames (int): One-sided size of context window to apply.

    Returns:
        An array of acoustic data that has been properly expanded. Every row will have grown by a factor of
        (1 + 2 * context_frames).
    """
    return np.column_stack([
        x[np.minimum(len(x) - 1, np.maximum(0, np.arange(len(x), dtype=np.int) + d))]
        for d in range(-context_frames, context_frames + 1)
    ])


def parse_script_line(script_line: str, script_path: str):
    """This function parses utterance specifications from a script file.

    It is expected the format obeys the following structure:
        (utterance name).feat=(feature filename)[(start frame), (end_frame)]

    If the (feature filename) starts with the magic string ".../", then this prefix will be replaced
    with the string represented by script_path. This functionality allows scripts and acoustic feature files
    to be agnostic to their absolute location in the filesystem.

    Args:
        script_line: The string to parse.
        script_path: The location of the script file that is being parsed.

    Returns:
        utt (string): A string identifier for the utterance.
        arc (string): The feature filename where the utterance can be found.
        frame_start (int): The offset of the utterance start in the arc.
        frame_end (int): The offset of the last utterance frame in arc.

    Note:
        The range [frame_start, frame_end] is inclusive.
    """
    m = re.match(r'(.*)\.feat=(.*)\[(\d+),(\d+)\]', script_line)
    assert(m)
    utt = m.group(1)
    arc = m.group(2)
    frame_start = int(m.group(3))
    frame_end = int(m.group(4))
    m = re.match(r'\.\.\.[/\\](.*)', arc)
    if m:
        arc = os.path.join(script_path, m.group(1))
    return utt, arc, frame_start, frame_end

def load_parameters(script_line: str, script_path: str):
    """This function extracts an utterance's string identifier and features from an archive.

    Args:
        script_line: The string representing the desired utterance.
        script_path: The location of the script file that contains the line.

    Returns:
        feat: An array representing the acoustic data, one row per frame of data.
        utt: A string identifier for the utterance.
    """
    utt, arc, frame_start, frame_end = parse_script_line(script_line, script_path)
    feat = read_htk_user_feat(arc)
    assert( frame_start == 0 )
    assert( frame_end + 1 - frame_start == len(feat))
    return feat, utt

def main():
    parser = argparse.ArgumentParser(description="Decode speech from parameter files.")
    parser.add_argument('-am', '--am', help='CNTK trained acoustic model', required=True, default=None)
    parser.add_argument('-decoding_graph', '--decoding_graph', help="Text-format openfst decoding graph", required=True, default=None)
    parser.add_argument('-label_map', '--label_map', help="Text files containing acoustic model state labels in the same order used when training the acoustic model", required=True, default=None)
    parser.add_argument('-scp', '--scp', help='Script file pointing to speech parameter files', required=True, default=None)
    parser.add_argument('-trn', '--trn', help='Filename to write output hypotheses', required=True, default=None)
    parser.add_argument('-lmweight', '--lmweight', help='Relative weight of LM score', required=False, type=float, default=30.0)
    parser.add_argument('-beam_width', '--beam_width', help='Maximum token count per frame', required=False, type=int, default=100)
    args = parser.parse_args()

    z = load_model(args.am)
    fst = FST(args.decoding_graph, args.label_map)

    script_path = os.path.split(args.scp)[0]
    time_start = time.time()
    frames_processed = 0

    try:
        with open(args.scp, 'r') as fscp:
            with open(args.trn, 'w', buffering=1) as ftrn:
                for line in fscp:
                    feats, utterance_name = load_parameters(line.rstrip(), script_path)
                    #feats = feature_stacker(feats)
                    activations = z.eval(feats.astype('f'))[0]
                    hypothesis = fst.decode(activations, beam_width=args.beam_width, lmweight=args.lmweight)
                    words = [
                        x for x in
                        map(operator.itemgetter(1), hypothesis)
                        if x not in ("<eps>", "<s>", "</s>")
                    ]
                    words.append('({})\n'.format(utterance_name))
                    ftrn.write(' '.join(words))
                    print(' '.join(words))
                    frames_processed += len(feats)


    except KeyboardInterrupt:
        print("[CTRL+C detected]")
    time_end = time.time()
    print('{:.1f} seconds, {:.2f} frames per second'.format(time_end-time_start, frames_processed / (time_end-time_start)))


def load_model(model_filename:str):
    """A helper function to load the acoustic model from disc.

    Args:
        model_filename (str): The file path to the acoustic model.
        """
    cntk_model = cntk.load_model(model_filename)

    #  First try and find output by name
    model_output = cntk_model.find_by_name('ScaledLogLikelihood')


    #  Fall back to first defined output
    if model_output is None:
        model_output = cntk_model.outputs[0]

    #  Create an object restricted to the desired output.
    cntk_model = cntk.combine(model_output)


    #  Optimized RNN models won't run on CPU without conversion.
    if 0 == cntk.use_default_device().type():
        cntk_model = cntk.misc.convert_optimized_rnnstack(cntk_model)

    return cntk_model


Token = namedtuple('token', 'id prev_id arc_number am_score lm_score')
Arc = namedtuple('arc', 'index source_state target_state ilabel olabel cost')

class token_manager:
    """This class encapsulates the token lifecycle, which makes the decoder class cleaner.
    """
    def __init__(self):

        # the tokens array holds all of the tokens we've committed to the search
        self.tokens = []

        # the first token should fall on the the start arc (number zero by convention)
        self.active_tokens = [Token(id=0, prev_id=-1, arc_number=0, am_score=0, lm_score=0)]

        # Tokens will be sequentially numbered with a unique integer ID. last_token_id stores the highest used ID.
        self.last_token_id = self.active_tokens[0].id


    def advance_token(self, prev_token: Token, next_arc, am_score, lm_score):
        """Create a successor of the given token, on the new arc, and additional scores.

        For proper book-keeping, all tokens but the initial token should be created with this function.

        :param prev_token: The predecessor for the new token.
        :param next_arc: The arc the new token occupies on the graph.
        :param am_score: The incremental acoustic model score associated with creating this token.
        :param lm_score: The incremental language model score associated with creating this token.
        :return: The new token.
        """
        self.last_token_id += 1
        return Token(
            self.last_token_id,
            prev_token.id,
            next_arc,
            prev_token.am_score + am_score,
            prev_token.lm_score + lm_score
        )

    def flatten_active_token_list(self, num_arcs: int, tok_list: List[Token]):
        """This helper function takes a list of tokens, and turns it into an equivalent set of sparse structures.

        This is necessary to enable the search to be coded against the sparse routines in the scipy library. It is
        assumed that each arc of the graph is associated with at most one token in the list.

        :param num_arcs: The length of the sparse vectors to create, usually the number of token-holding arcs in
        the WFST underlying the search.
        :param tok_list: A list of tokens to convert.
        :return: score, token_index. score is a sparse column vector with one entry for every token. token_index
        is a mapping from the row index of score, to the list index of the token that created it.
        """

        # token's score
        v = np.array([x.am_score + x.lm_score for x in tok_list], dtype=np.float32)
        v = np.exp(v - np.max(v))
        # make a column vector; row index is arc number
        r = np.array([x.arc_number for x in tok_list], dtype=np.int)
        c = np.zeros(r.shape)
        score = scipy.sparse.csc_matrix(
            (v, (r, c)),
            shape=(num_arcs,1),
            dtype=np.float32
        )

        token_index = np.ones(num_arcs, dtype=np.int32)*-1  # bogus initialization
        for i, x in enumerate(tok_list):
            token_index[x.arc_number] = i

        return score, token_index

    def tok_backtrace(self, looking_for_tokid=None):
        """This function finds the best path described by the tokens created so far.

        The candidates for ultimate token in the path are taken from the self.active_tokens array. Starting
        with the best token in this array, it finds each token's predecessor until landing on the initial token.

        :return: A sequence of arc numbers that represent discovered path.
        """

        if looking_for_tokid is None:
            looking_for_tokid = max(self.active_tokens, key=lambda x: x.am_score + x.lm_score).id

        path=[]
        for tok in (self.tokens + self.active_tokens)[::-1]:  # search backward through tokens
            if tok.id == looking_for_tokid:
                arc_number = tok.arc_number
                path.append(arc_number)
                looking_for_tokid = tok.prev_id
        # reverse backtrace so tokens are in forward-time-order
        path = path[::-1]

        # Combine sequences of identical arcs into one representative arc_number
        segments = [k for k, x in itertools.groupby(path)]

        return segments

    def commit_active_tokens(self):
        """Add the current active_tokens to the set of tokens to retain.

        This is often called just before modifying (updating) the set of tokens.
        """
        self.tokens += self.active_tokens

    def beam_prune(self, beam_width: int):
        """
        A simple beam pruning function.

        This function eliminates, from the active_tokens list, any tokens that fall outside of the given
        pruning beam width.

        :param beam_width: The number of tokens to retain.
        """
        if len(self.active_tokens) > beam_width:
            self.active_tokens = sorted(
                self.active_tokens, key=lambda x: x.am_score + x.lm_score, reverse=True
            )[0:beam_width]


class FST:
    def __init__(self, fst_file: str, label_mapping: str=None):
        """
        This class encapsulates the loading and processing of WFST decoding graphs.

        :param fst_file: The text-format WFST representing the decoding graph.
        :param label_mapping: A mapping from acoustic model label strings to label indices, identical to one
        that was used when training the acoustic model.
        """
        self._arcs = []
        self._final = {}
        self._index2label = []
        self._label2index = {}

        self._load_map(label_mapping)
        self._load_fst(fst_file)

    def _preprocess_activations(self, act):
        """This function renormalizes acoustic model scores to be in a reasonable range.

        It ensures that the maximum score for any time is equal to zero by adding a bias to every row. This
        does not affect the relative weight of any path, but does keep the path scores in a reasonable range
        and prevents overflow or underflow in some situations.

        It also scales the acoustic model scores. This scaling is common and is necessary to balance the
        information provided by the acoustic model and language model.

        :param act: The raw score output from the acouatic model.
        :return: Renormalized activations.
        """
        return (act - np.max(act, axis=1).reshape((act.shape[0], 1)))


    def decode(self, act, beam_width, lmweight, alignment: List[str]=None, ftrans=None, etrans=None, strans=None):
        """
        Find the best path through the decoding graph, using a given set of acoustic model scores.

        The ability to specify per-utterance transition structures allows for the possibility of writing
        pseudo-alignment algorithms, which are not yet a part of the current course, but the instructor found
        useful for debugging the code.

        :param act: The acoustic model scores.
        :param ftrans: A sparse matrix representing linear probabilities of non-epsilon graph transitions.
        :param etrans: A sparse matrix representing linear probabilities of epsilon graph transitions.
        :param strans: A sparse matrix representing log probabilities of all graph transitions.
        :return: A list of (input label, output_label) pairs representing the decoded path.
        """

        if alignment is not None:
            alignment = [self._label2index[x] for x in alignment]

        #  If the caller doesn't give us transition matrices, then use our own.
        if ftrans is None:
            ftrans = self.emit_trans
        if etrans is None:
            etrans = self.eps_trans
        if strans is None:
            strans = self.log_score

        #  Turn the given acoustic model scores into something useful
        act = self._preprocess_activations(act) / lmweight

        #  Create an utterance-specific token manager to hold the state of the search
        tm = token_manager()

        def do_forward(tok_list, transition_matrix, obs_vector=None):
            """Implements the search-update algorithm using sparse matrix-vector primitives
            """

            #  Convert the token list into a sparse structure
            src_score, src_token_index = tm.flatten_active_token_list(len(self._arcs), tok_list)

            # Project the tokens forward through the given transition matrix. Note that this is not
            # a matrix multiplication, but an application of the previous tokens to the columns of the
            # transition matrix. The (i,j) element in the resulting two-dimensional structure represents
            # the score assocaited with creating a new token on arc i, from an old token on arc j.
            trans = transition_matrix.multiply(src_score.T)

            # Convert the sparse trans matrix into two obects: row_to_column, which for every row of
            # trans, indicates which column had the best score, and active_rows, which is the set
            # of rows in trans with non-zero entries. This tells us which tokens will be created
            # (active_rows), and who their best predecessor is (row_to_column).
            row_to_column = np.array(trans.argmax(axis=1)).squeeze()
            active_rows = trans.max(axis=1).nonzero()[0]

            # Create a complete set of new tokens and return it.
            new_tok = [
                tm.advance_token(
                    tok_list[src_token_index[row_to_column[r]]],
                    r,
                    obs_vector[self._arcs[r].ilabel] if obs_vector is not None else 0,
                    strans[r, row_to_column[r]]
                )
                for r in active_rows
            ]
            return new_tok

        # Here is the core of the search algorithm. It loops over time, using the acoustic model scores
        for t, obs in enumerate(act):
            # Store any tokens created from the previous time frame
            tm.commit_active_tokens()

            # Transform all of the new tokens onto their succeeding non-epsilon arcs.
            tm.active_tokens = do_forward(tm.active_tokens, ftrans, np.array(obs).squeeze())

            # Keep the search to a manageable size.
            tm.beam_prune(beam_width)

            # Advance the tokens we've just created onto any arcs with epsilon input symbols they can reach.
            epsilon_tokens = []
            prev_tokens = tm.active_tokens
            while len(prev_tokens)>0:
                prev_tokens = do_forward(prev_tokens, etrans)
                epsilon_tokens += prev_tokens

            # Among the epsilon tokens we've just created, only keep the best for each arc.
            epsilon_tokens = [
                max(x, key=lambda token: token.am_score + token.lm_score)
                for k, x in itertools.groupby(
                    sorted(
                        epsilon_tokens,
                        key=lambda token: token.arc_number
                    ),
                    key=lambda token: token.arc_number)
            ]

            # Ensure the tokens are sorted by ID. This invariant is used by the tok_backtrace member function.
            epsilon_tokens.sort(key=lambda token: token.id)

            # Now, tm.active_tokens includes all tokens created for this time step. These have all consumed
            # exactly (t+1) symbols on the input side.
            tm.active_tokens += epsilon_tokens

        tm.commit_active_tokens()

        # apply final state scores
        dest_state = [self._arcs[x.arc_number].target_state for x in tm.active_tokens]
        tm.active_tokens = [
            Token(x.id, x.prev_id, x.arc_number, x.am_score, x.lm_score - self._final[s])
            for x, s in zip(tm.active_tokens, dest_state) if s in self._final
        ]

        best_tok = max(tm.active_tokens, key=lambda x: x.am_score + x.lm_score)
        print(
            "best cost: AM={} LM={} JOINT={}".format(
                best_tok.am_score, best_tok.lm_score, best_tok.am_score + best_tok.lm_score
            )
        )

        # return best path
        return  map(lambda arc: (self._index2label[arc.ilabel], arc.olabel),
            map(lambda arc_number: self._arcs[arc_number],
                tm.tok_backtrace()
            )
        )

    def _load_map(self, filename):
        """Read the label mapping file from disc.

        This mapping is used to associate strings of the deconding graph's input label with the index into
        acosutic model score vectors.
        """
        with open(filename) as f:
            self._index2label = [
                '[' + x.rstrip().replace('.', '_') + ']' for x in f
            ]
        self._label2index = {'<s>': -1, '<eps>': -2}
        for i, x in enumerate(self._index2label):
            self._label2index[x] = i
        self._index2label += ['<eps>', '<s>']

    def _load_fst(self, filename):
        """Read a text-format WFST decoding graph into memory.
        """
        arcout = []
        self._final = {}
        self._arcs = []

        # this start-arc is where every token lives before the first frame of data
        self._arcs.append(Arc(0, -1, 0, self._label2index['<eps>'], '<eps>', float(0)))

        # Read the FST into our self_.arcs list. Specialized functions parse "final state" and "normal arc"
        # lines of the input file, keyed on the number of space-separated fields in the line.
        def process_final_state(parts):
            assert (len(parts) in (1, 2))
            self._final[int(parts[0])] = float(parts[1] if len(parts) > 1 else 0)
        def process_normal_arc(parts):
            assert (len(parts) in (4, 5))
            # state numbers -> integers
            sorc = int(parts[0])
            dest = int(parts[1])
            ilab = self._label2index[parts[2]]
            olab = parts[3]
            cost = float(parts[4] if len(parts) > 4 else 0)
            # save this arc
            self._arcs.append(Arc(len(self._arcs), sorc, dest, ilab, olab, cost))
        with open(filename) as f:
            for line in f:
                parts = line.rstrip().split()
                if len(parts) <= 2:
                    process_final_state(parts)
                else:
                    process_normal_arc(parts)

        # Pre-index all arcs coming out of a state to speed up transition-matrix creation.
        arcout = [() for _ in range(1 + max(x.source_state for x in self._arcs))]
        for source_state, arcs in itertools.groupby(
                sorted(self._arcs, key=lambda arc: arc.source_state),
                key= lambda arc: arc.source_state):
            arcout[source_state] = [arc.index for arc in arcs]


        # We encode the graph transition structure as three sparse matrices. Each i,j element represents a
        # cost for a path through the graph to transition from arc number i to arc number j. The log_score
        # matrix faithfully represents these scores, whereas the emit_trans and eps_trans matrices store the
        # scores processed through exp(). This allows for our implementation to use standard sparse
        # matrix-vector operations to efficiently compute path scores. The emit_trans contains only nonzero rows
        # that don't have epsilons on the corresponding arcs, and the eps_trans contains the rows that do.

        emit_row, emit_col, emit_val = [], [], []
        eps_row, eps_col, eps_val = [], [], []

        for arc in self._arcs:
            if arc.ilabel >= 0:
                # non-epsilon -> implies zero-cost self-loop
                emit_col.append(arc.index)
                emit_row.append(arc.index)
                emit_val.append(float(0))
            next_state = arc.target_state
            for next_arc_index in arcout[next_state]:
                next_arc = self._arcs[next_arc_index]
                score = -next_arc[-1]
                if next_arc.ilabel >= 0:
                    # non-epsilon row
                    emit_col.append(arc.index)
                    emit_row.append(next_arc_index)
                    emit_val.append(score)
                else:
                    # epsilon row
                    eps_col.append(arc.index)
                    eps_row.append(next_arc_index)
                    eps_val.append(score)

        # The linear transition score for arcs with emitting symbols
        self.emit_trans = scipy.sparse.csr_matrix(
            (np.exp(emit_val), (emit_row, emit_col)),
            shape=(len(self._arcs), len(self._arcs)),
            dtype=np.float32
        )
        # The linear transition score for arcs with non-emitting symbols
        self.eps_trans = scipy.sparse.csr_matrix(
            (np.exp(eps_val), (eps_row, eps_col)),
            shape=(len(self._arcs), len(self._arcs)),
            dtype=np.float32
        )
        # The log-score for all arcs
        self.log_score = scipy.sparse.csr_matrix(
            (emit_val + eps_val, (emit_row + eps_row, emit_col + eps_col)),
            shape=(len(self._arcs), len(self._arcs)),
            dtype=np.float32
        )

if __name__ == '__main__':
    main()
