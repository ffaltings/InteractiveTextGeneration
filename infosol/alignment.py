import random
import itertools
import torch
import numpy as np

from transformers import AutoTokenizer, BertModel

def find_max_score(alignment_matrix, score_matrix, i, j, baseline_score=0):
    """
    Maximum alignment score up to this position, given max alignment scores up to previous positions
    """
    del_score = alignment_matrix[i, j-1] + baseline_score
    ins_score = alignment_matrix[i-1, j] + baseline_score
    match_score = alignment_matrix[i-1,j-1] + score_matrix[i-1, j-1]
    scores = (ins_score, del_score, match_score) # this order matters for priority b/w insert/del
    return np.max(scores), np.argmax(scores)

def compute_alignment_scores(score_matrix, baseline_score=0):
    """
    Compute maximum alignment score for each position using DP
    """
    alignment_matrix = np.zeros((score_matrix.shape[0] + 1, score_matrix.shape[1] + 1))
    for i in range(1, alignment_matrix.shape[0]):
        alignment_matrix[i,0] = i * (baseline_score*10)/10 #avoids a weird bug with floating point precision
    for j in range(1, alignment_matrix.shape[1]):
        alignment_matrix[0,j] = j * (baseline_score*10)/10
    move_matrix = np.zeros(alignment_matrix.shape)
    for j in range(1,alignment_matrix.shape[1]):
        for i in range(1,alignment_matrix.shape[0]):
            alignment_matrix[i,j], move_matrix[i,j] = find_max_score(alignment_matrix, score_matrix, i, j, baseline_score)
    return alignment_matrix, move_matrix

def get_alignment(move_matrix, tokens_a, tokens_b, score_matrix=None, baseline_score=0.):
    """
    Retrieve the actual alignment from the matrix of alignment scores,
    and matrix of "moves" thru the score matrix
    """
    i,j = move_matrix.shape
    i -= 1
    j -= 1
    alignment = []
    alignment_score = []
    while i >= 1 and j >= 1:
        move = move_matrix[i,j]
        if move == 2:
            op = (tokens_a[i-1], tokens_b[j-1])
            if score_matrix is not None:
                alignment_score.append(score_matrix[i-1,j-1])
#                op = op + (float(score_matrix[i-1, j-1]),)
            alignment.append(op)
            i -= 1
            j -= 1
        elif move == 0:
            op = (tokens_a[i-1], '')
            if score_matrix is not None:
#                op = op + (baseline_score,)
                alignment_score.append(baseline_score)
            alignment.append(op)
            i -= 1
        elif move == 1:
            op = ('',tokens_b[j-1])
            if score_matrix is not None:
#                op = op + (baseline_score,)
                alignment_score.append(baseline_score)
            alignment.append(op)
            j -= 1
    if i == 0:
        while j >= 1:
            op = ('', tokens_b[j-1])
            if score_matrix is not None:
#                op = op + (baseline_score,)
                alignment_score.append(baseline_score)
            alignment.append(op)
            j -= 1
    elif j == 0:
        while i >= 1:
            op = (tokens_a[i-1], '')
            if score_matrix is not None:
#                op = op + (baseline_score,)
                alignment_score.append(baseline_score)
            alignment.append(op)
            i -= 1
    return alignment[::-1], alignment_score[::-1]

def batch_cos_sim(tokens_a, tokens_b, model, tokenizer, device=torch.device('cpu'), use_model=True):
    """
    Compute cosine similarities
    """
    if len(tokens_a) != len(tokens_b):
        raise ValueError('batch sizes cannot be different')
    for tokens in tokens_a + tokens_b:
        if len(tokens) == 0:
            raise ValueError('tokens cannot be empty')

    def batch_encodings(batch):
        # convert to ids
        batch = [torch.tensor(
            tokenizer.convert_tokens_to_ids(tokens),
            dtype=torch.long,
            device=device) for tokens in batch]
        max_len = np.max([ids.shape[0] for ids in batch])
        input_ids = torch.zeros((len(batch), max_len), dtype=torch.long,
                               device=device)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.int32)
        for i,b in enumerate(batch):
            input_ids[i, :len(b)] = b
            attention_mask[i, :len(b)] = 1
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)

        for i in range(len(batch)):
            canv_len = attention_mask[i,:].sum()
            yield out.last_hidden_state[i, :canv_len],\
                input_ids[i, :canv_len]

    batch_enc_a = batch_encodings(tokens_a)
    batch_enc_b = batch_encodings(tokens_b)

    for (enc_a, ids_a), (enc_b, ids_b) in zip(batch_enc_a,
                                             batch_enc_b):
        exact_matches = (ids_a.unsqueeze(-1).expand(-1, ids_b.shape[0])\
                        == ids_b.unsqueeze(0).expand(ids_a.shape[0], -1)).type(torch.float)

        if not use_model:
            yield exact_matches

        dot_products = torch.matmul(enc_a, enc_b.transpose(0,1))
        try:
            norms = torch.matmul(torch.norm(enc_a, dim=-1).unsqueeze(1),
                                torch.norm(enc_b, dim=-1).unsqueeze(0))
        except IndexError as e:
            print(enc_a.shape, enc_b.shape, ids_a, ids_b)
            raise e
        yield torch.max(dot_products/norms, exact_matches)

def batch_align(tokens_a, tokens_b, model, tokenizer,
    baseline_score=0., device=torch.device('cpu'), **kwargs):
    """
    Align tokens
    :param tokens_a: list of lists of tokens
    :param tokens_b: list of lists of tokens
    :param model: encoder model
    :param tokenizer: tokenizer
    :param baseline_score: score assigned for non matches
    """
    if len(tokens_a) != len(tokens_b):
        raise ValueError('batch sizes cannot differ')
    non_empty_idxs = [i for i,(tok_a,tok_b) in enumerate(
        zip(tokens_a, tokens_b)) if len(tok_a) > 0 and len(tok_b) > 0]
    score_matrices = batch_cos_sim([tokens_a[i] for i in non_empty_idxs],
        [tokens_b[i] for i in non_empty_idxs], model, tokenizer, device=device, **kwargs)

    for j in range(len(tokens_a)):
        tok_a = tokens_a[j]
        tok_b = tokens_b[j]
        if len(tok_a) == 0:
            alignment = [('', b) for b in tok_b]
            scores = [baseline_score] * len(alignment)
            yield Alignment(alignment, scores=scores, baseline_score=baseline_score)
        elif len(tok_b) == 0:
            alignment = [(a, '') for a in tok_a]
            scores = [baseline_score] * len(alignment)
            yield Alignment(alignment, scores=scores, baseline_score=baseline_score)
        else:
            score_matrix = next(score_matrices).cpu().numpy()
            alignment_matrix, move_matrix = compute_alignment_scores(
                score_matrix, baseline_score=baseline_score)
            alignment, scores = get_alignment(move_matrix, tok_a, tok_b,
                                     score_matrix=score_matrix,
                                     baseline_score=baseline_score)
            yield Alignment(alignment, scores=scores, baseline_score=baseline_score)

def align(tokens_a, tokens_b, model, tokenizer, **kwargs):
    return next(batch_align([tokens_a], [tokens_b], model, tokenizer, **kwargs))

def batch_align_canvases(canvases, targets, model, tokenizer, **kwargs):
    """
    Align canvases to target tokens, differs from aligning tokens because it handles the type ids of the canvases
    """
    solid_tokens = [canvas.clean().tokens for canvas in canvases]
    for canvas,alignment in zip(canvases,
        batch_align(solid_tokens, targets, model, tokenizer, **kwargs)):
        recovered_alignment = []
        recovered_alignment_scores = []
        i,j = 0,0
        while i < len(canvas) or j < len(alignment):
            if i < len(canvas) and j < len(alignment) and \
               canvas.tokens[i] == alignment.alignment[j][0] and\
               not canvas.is_stricken(i) and not alignment.is_insertion(j):
                recovered_alignment.append(alignment.alignment[j])
                recovered_alignment_scores.append(alignment.scores[j])
                i += 1
                j += 1
            elif i < len(canvas) and canvas.is_stricken(i):
                recovered_alignment.append((canvas.tokens[i], ''))
                recovered_alignment_scores.append(1)
                i += 1
            elif j < len(alignment) and alignment.is_insertion(j):
                recovered_alignment.append(alignment.alignment[j])
                recovered_alignment_scores.append(alignment.scores[j])
                j += 1
            else:
                print(canvas.tokens, canvas.type_ids)
                print(alignment)
                print(i,j)
                print(recovered_alignment)
                print(canvas.tokens[i], canvas.type_ids[i], alignment.alignment[j])
                raise RuntimeError('Error with alignment')
        alignment = Alignment(recovered_alignment, recovered_alignment_scores)
        alignment.set_type_ids(canvas.type_ids)
        yield alignment

def align_canvas(canvas, target, model, tokenizer, **kwargs):
    return next(batch_align_canvases([canvas], [target], model, tokenizer, **kwargs))

class Canvas():

    """
    Canvas is a sequence of tokens with type ids
    """

    html_types = [
        "{}", # 0: plain text
        "<ins>{}</ins>", #1: agent inserted text
        "<ins style='color:blue'>{}</ins>", #2 oracle inserted text
        "<del>{}</del>", #3: user deleted text
        "<del style='color:blue'>{}</del>", #4 oracle deleted text
    ]

    @staticmethod
    def latex_type(tid, tok):
        """
        Utility for rendering to latex
        """
        if tid == 0:
            text = tok
        elif tid == 1:
            text = ''.join(('\\textcolor{seagreen}{',tok,'}'))
        elif tid == 2:
            text = ''.join(('\\textcolor{BrickRed}{',tok,'}'))
        elif tid == 3:
            text = ''.join(('\sout{', tok, '}'))
        elif tid == 4:
            text = ''.join(('\sout{\\textcolor{BrickRed}{', tok, '}}'))
        if tok[0] == ' ' and tid != 0:
            text = ' '+text
        return text

    def __init__(self, tokens, type_ids=None):
        if not type_ids is None and len(tokens) != len(type_ids):
            raise ValueError('length of tokens must match length of type ids')
        self.tokens = tokens
        if type_ids is None:
            type_ids = [0] * len(tokens)
        self.type_ids = type_ids

    def __len__(self):
        return len(self.tokens)

    def __eq__(self, other):
        return self.tokens == other.tokens and self.type_ids == other.type_ids

    def __lt__(self, other):
        return len(self.tokens) < len(other.tokens)

    def __gt__(self, other):
        return len(self.tokens) > len(other.tokens)

    def __le__(self, other):
        return len(self.tokens) <= len(other.tokens)

    def __ge__(self, other):
        return len(self.tokens) >= len(other.tokens)

    def __repr__(self):
        return str((self.tokens, self.type_ids))

    def is_stricken(self, i):
        return self.type_ids[i] >= 3

    def operate(self, loc, operation, token, agent=0):
        """
        Apply an edit to a canvas
        """
        if operation == 0: # insertion
            self.tokens.insert(loc+1, token)
            self.type_ids.insert(loc+1, agent+1)
        elif operation == 1: # substitution
            self.tokens[loc] = token
            self.type_ids[loc] = agent+1
        elif operation == 2: # deletion
            if self.type_ids[loc] != 0:
                del self.tokens[loc]
                del self.type_ids[loc]
            else:
                self.type_ids[loc] = agent + 3
        else:
            raise ValueError('Invalid Operation: {}'.format(operation))

    def render(self, tokenizer, clean=True):
        """
        Return string from the canvas
        """
        if clean:
            canvas = self.clean()
        if len(canvas) == 0: return ''
        else:
            return tokenizer.convert_tokens_to_string(canvas.tokens)

    def render_to_html(self, tokenizer):
        """
        Format into html
        """
        html_string = ''
        for tok, tid in zip(self.tokens, self.type_ids):
            html_string += Canvas.html_types[tid].format(
                tokenizer.convert_tokens_to_string(tok))
        return html_string

    def render_to_latex(self, tokenizer):
        """
        Format into LATEX
        """
        latex_string = ''
        for tok, tid in zip(self.tokens, self.type_ids):
            latex_string += Canvas.latex_type(
                tid, tokenizer.convert_tokens_to_string(tok))
        return latex_string

    def clean(self):
        """
        Remove tokens that are stricken out and remove type ids (except for oracle type ids)
        """
        tokens = [t for i,t in enumerate(self.tokens) if not self.is_stricken(i)]
        type_ids = [tid for i,tid in enumerate(self.type_ids) if not self.is_stricken(i)]
        # reset the type ids for model insertions
        type_ids = [0 if tid==1 else tid for tid in type_ids]
        return Canvas(tokens, type_ids)

    def copy(self):
        return Canvas(list(self.tokens), list(self.type_ids))

class Alignment():

    """
    Alignment between two sets of tokens
    """

    def __init__(self, alignment, scores=None, type_ids=None,
                 baseline_score=0.3, add_type_ids=True):
        self.alignment=alignment

        if not scores is None and len(scores) != len(alignment):
            raise ValueError('length of scores must match length of alignment')
        self.scores=scores
        self.baseline_score = baseline_score

        if not type_ids is None:
            self.set_type_ids(type_ids)
        elif add_type_ids:
            self.set_type_ids([0] * len(self.get_source_tokens()))
        else:
            self.type_ids = None

    def copy(self):
        return Alignment(list(self.alignment), list(self.scores))

    def __len__(self):
        return len(self.alignment)

    def __str__(self):
        print_str = ''
        if len(self.alignment) == 0: return print_str
        src_pad = max(1, np.max([len(tup[0]) for tup in self.alignment]))
        tgt_pad = max(1, np.max([len(tup[1]) for tup in self.alignment]))
        for i in range(len(self.alignment)):
            line = '{src:{src_pad}} - {tgt:{tgt_pad}}'.format(src=self.alignment[i][0],
                tgt=self.alignment[i][1], src_pad=src_pad, tgt_pad=tgt_pad)
            if not self.scores is None:
                line += ' {:.2f}'.format(self.scores[i])
            if not self.type_ids is None:
                line += ' {:2}'.format(self.type_ids[i])
            print_str += line + '\n'
        return print_str

    def is_match(self, idx):
        return self.alignment[idx][0] != '' and self.alignment[idx][1] != ''

    def is_exact_match(self, idx):
        return self.alignment[idx][0] == self.alignment[idx][1]

    def is_insertion(self, idx):
        return self.alignment[idx][0] == '' and self.alignment[idx][1] != ''

    def is_addition(self, idx):
        return self.alignment[idx][1] != '' and self.alignment[idx][0] != self.alignment[idx][1]

    def is_deletion(self, idx):
        return self.alignment[idx][0] != '' and self.alignment[idx][1] == ''

    def get_non_const_ops(self):
        """
        All positions in alignment that aren't exact matches
        """
        if self.type_ids is None:
            return [i for i in range(len(self.alignment))\
                if not self.is_exact_match(i)]
        else:
            return [i for i in range(len(self.alignment))\
                if not self.is_exact_match(i) and self.type_ids[i] < 3]

    def get_adjacent_ops(self, idxs, adj_range = 1):
        """
        Return idxs of adjacent ops in alignment. Adjacent idxs are idxs 
        that are near a pair of matched tokens (not necessarily an exact match)
        """
        def is_adjacent(idx, adj_range):
            min_lim = max(0, idx - adj_range)
            max_lim = min(len(self.alignment), idx + adj_range + 1)
            for i in range(min_lim, max_lim):
                if self.is_match(i):
                    return True
            return False
        ret_idxs = [i for i in idxs if is_adjacent(i, adj_range)]
        if not ret_idxs: return idxs
        return ret_idxs

    def get_actions(self, action_idxs, include_deleted_words=False):
        """
        Get actual action tuples for given idxs of an alignment
        """
        actions = {}
        j = -1
        k = -1# <- keep track of this for insertions, -1 is sentinel token
        for i in range(len(self.alignment)):
            if not self.is_insertion(i):
                j += 1
                if self.type_ids is None:
                    k = j
                else:
                    if self.type_ids[i] < 3:
                        k = j
            if i in action_idxs:
                if self.is_insertion(i):
                    actions[i] = (k, 0, self.alignment[i][1])
                elif self.is_deletion(i):
                    if include_deleted_words:
                        actions[i] = (j, 2, None, self.alignment[i][0])
                    else:
                        actions[i] = (j, 2, None)
                else: # substitution
                    actions[i] = (j, 1, self.alignment[i][1])
        actions = [actions[i] for i in action_idxs]
        return actions

    def push_forward(self, idxs, agent = 0):
        """
        Push alignment forward at given idxs (i.e. replace the source token with the target token at the location
        giving an exact match)
        """
        if len(idxs) == 0: return
        if np.min(idxs) < 0 or np.max(idxs) >= len(self.alignment):
            raise ValueError('idxs out of range')
        delete_idxs = []
        for i in idxs:
            if self.type_ids is None:
                if not self.is_deletion(i):
                    self.alignment[i] = (self.alignment[i][1], self.alignment[i][1])
                else:
                    delete_idxs.append(i)
            else:
                if not self.is_deletion(i):
                    self.alignment[i] = (self.alignment[i][1], self.alignment[i][1])
                    self.type_ids[i] = agent+1
                elif self.type_ids[i] == 0:
                    self.type_ids[i] = agent+3
                elif self.type_ids[i] < 3: #otherwise token was already deleted
                    delete_idxs.append(i)
        self.alignment = [tup for i,tup in enumerate(self.alignment) if i not in delete_idxs]
        if not self.scores is None:
            self.scores = [s for i,s in enumerate(self.scores) if i not in delete_idxs]
        if not self.type_ids is None:
            self.type_ids = [tid for i,tid in enumerate(self.type_ids) if i not in delete_idxs]

    def canvas_location_to_alignment_location(self, canvas_location):
        """
        Map a location in the source canvas to the alignment
        """
        j = 0
        for i in range(len(self.alignment)):
            if not self.is_insertion(i):
                canvas_location -= 1
                if canvas_location < -1: break
                j = i
        return j

    def operate(self, canvas_location, operation, token, agent = 0):
        '''
        Operate on the source canvas while updating the alignment to the target tokens
        '''
        if canvas_location >= len(self.get_source_tokens()):
            raise ValueError('canvas location out of bounds')
        if operation == 0: #insertion
            if token is None: raise ValueError('token cannot be None when inserting')
            # insert on the left of the next token to keep contiguous insertions and deletions grouped
            location = self.canvas_location_to_alignment_location(canvas_location+1)
            self.alignment.insert(location, (token, ''))
            if not self.scores is None:
                self.scores.insert(location, self.baseline_score)
            if not self.type_ids is None:
                self.type_ids.insert(location, agent+1)
        elif operation == 1: #substitution
            if token is None: raise ValueError('token cannot be None when substituting')
            if canvas_location == -1: raise ValueError('cannot substitute bos token')

            location = self.canvas_location_to_alignment_location(canvas_location)
            if not self.type_ids is None and self.type_ids[location] >= 3:
                raise ValueError('cannot edit deleted token')
            if not self.type_ids is None: self.type_ids[location] = 1 # trick to make deletion operate correctly
            self.operate(canvas_location, 2, token, agent)
            self.operate(canvas_location-1, 0, token, agent)
        elif operation == 2: #deletion
            location = self.canvas_location_to_alignment_location(canvas_location)
            if not self.type_ids is None and self.type_ids[location] >= 3:
                raise ValueError('cannot edit deleted token')
            if self.is_match(location):
                deleted_token = self.alignment[location][0]
                self.alignment[location] = ('', self.alignment[location][1])
                if not self.scores is None:
                    self.scores[location] = self.baseline_score
                if not self.type_ids is None and self.type_ids[location] == 0: #this needs to remain
                    self.type_ids[location] = -1
                    replacement_location = self.canvas_location_to_alignment_location(
                        canvas_location)
                    self.alignment.insert(replacement_location, (deleted_token, ''))
                    self.type_ids.insert(replacement_location, agent+3)
                    if not self.scores is None:
                        self.scores.insert(replacement_location, self.baseline_score)
                elif not self.type_ids is None: self.type_ids[location] = -1
            else:
                if not self.type_ids is None and self.type_ids[location] == 0:
                    self.type_ids[location] = agent+3
                else:
                    del self.alignment[location]
                    if not self.scores is None:
                        del self.scores[location]
                    if not self.type_ids is None:
                        del self.type_ids[location]

        if len(self.get_type_ids()) != len(self.get_source_tokens()):
            print(canvas_location, operation, token)
            print(self.get_type_ids(), self.get_source_tokens())
            print(self)
            raise RuntimeError('violated consitency')

    def set_type_ids(self, type_ids):
        '''
        Add type ids to alignment
        :param type_ids: type ids to add. Assumes they correspond to the type ids of the source tokens
        '''
        if len(type_ids) != len(self.get_source_tokens()):
            raise ValueError('length of type ids must match length of source tokens in alignment')
        self.type_ids = []
        j = 0
        for i in range(len(self.alignment)):
            if not self.is_insertion(i):
                tag = type_ids[j]
                j += 1
            else: tag = -1
            self.type_ids.append(tag)

    def get_type_ids(self):
        if self.type_ids is None: return None
        else: return [tid for tid in self.type_ids if tid != -1]

    def get_source_tokens(self):
        return [self.alignment[i][0] for i in range(len(self.alignment))\
            if not self.is_insertion(i)]

    def get_target_tokens(self):
        return [self.alignment[i][1] for i in range(len(self.alignment))\
            if not self.is_deletion(i)]

    def get_source_canvas(self):
        source_tokens = self.get_source_tokens()
        type_ids = self.get_type_ids()
        return Canvas(source_tokens, type_ids)

    def get_target_canvas(self):
        target_tokens = self.get_target_tokens()
        return Canvas(target_tokens)

def print_alignment(alignment):
    for (a,b) in alignment:
        print("{} - {}".format(a,b))

def get_frozen_idxs(alignment):
    """
    Determine idxs with exact matches. Opposite of get_non_const_ops
    """
    idxs = []
    i = 0
    for tup in alignment:
        if tup[0] == tup[1]:
            idxs.append(i)
        if tup[0] != '':
            i += 1
    return idxs

def score_token_idf(token, idf_dict, max_score):
    return idf_dict[token] if token in idf_dict else max_score

def get_non_const_ops(alignment):
    return [i for i,op in enumerate(alignment) if op[0] != op[1] and op[2] < 3]

def get_adjacent_ops(idxs, alignment, adj_range = 1):
    """
    Return idxs of adjacent ops in alignment. Adjacent idxs are idxs 
    that are near a pair of matched tokens (not necessarily an exact match)
    """
    def is_adjacent(idx, adj_range):
        if idx < adj_range: # "adjacent to start token"
            return True
        min_lim = max(0, idx - adj_range)
        max_lim = min(len(alignment), idx + adj_range + 1)
        for i in range(min_lim, max_lim):
            if alignment[i][0] != '' and alignment[i][1] != '': # matched tokens?
                return True
        return False
    ret_idxs = [i for i in idxs if is_adjacent(i, adj_range)]
    return ret_idxs

def operate_canvas(canvas, token_type_ids, loc, operation, token, agent=0):
    """
    Apply an edit to a canvas
    """
    canvas = list(canvas)
    token_type_ids = list(token_type_ids)
    if operation == 0: # insertion
        canvas = canvas[:loc+1] + [token] + canvas[loc+1:]
        token_type_ids = token_type_ids[:loc+1] + [agent+1] + token_type_ids[loc+1:]
    elif operation == 1: # substitution
        canvas = canvas[:loc] + [token] + canvas[loc+1:]
        token_type_ids = token_type_ids[:loc] + [agent+1] + token_type_ids[loc+1:]
    elif operation == 2: # deletion
        if token_type_ids[loc] != 0:
            canvas = canvas[:loc] + canvas[loc+1:]
            token_type_ids = token_type_ids[:loc] + token_type_ids[loc+1:]
        else:
            token_type_ids[loc] = agent + 3
    else:
        raise ValueError('Invalid Operation: {}'.format(operation))
    return canvas, token_type_ids

def operate_tagged_alignment(alignment, idxs, agent = 0):
    """
    Push alignment forward at given idxs
    :param alignment: tagged alignment
    """
    alignment = list(alignment) #so it doesn't do anything in place
    for i in idxs:
        if alignment[i][1] != '':
            alignment[i] = (alignment[i][1], alignment[i][1], agent + 1)
        elif alignment[i][-1] == 0:
            alignment[i] = (alignment[i][0], alignment[i][1], agent + 3)
        else:
            alignment[i] = (alignment[i][0], alignment[i][1], -2)
    alignment = [tup for tup in alignment if tup[-1] >= -1]
    return alignment

def compress_alignment(alignment, alignment_scores):
    # TODO: this should assume that the alignment can contain other tags, such as scores
    new_alignment, new_alignment_scores = [], []
    for tup, score in zip(alignment, alignment_scores):
        if tup[0] == '' and tup[1] == '': continue
        new_alignment.append(tup)
        new_alignment_scores.append(score)
    return new_alignment, new_alignment_scores

def update_idxs(idxs, loc, op, tok):
    """
    Update a set of idxs given an action that has been 
    applied to a canvas
    """
    new_idxs = []
    for i in list(idxs):
        if i > loc: #unaffected otherwise
            if op == 0:
                i += 1
            elif op == 2:
                i -= 1
        new_idxs.append(i)
    return set(new_idxs)

def get_actions_from_tagged_alignment(alignment, action_idxs, include_deleted_words=False):
    """
    Get actual action tuples for given idxs of an alignment
    """
    input_tokens = []
    for i,tup in enumerate(alignment):
        input_tokens.append(tup[0])

    actions = {}
    j = -1
    k = -1# <- keep track of this for insertions, -1 is sentinel token
    for i, (tok, op) in enumerate(zip(input_tokens, alignment)):
        if tok != '' and op[2] < 3:
            j += 1
            k = j
        elif tok != '' and op[2] >= 3:
            j += 1
        if i in action_idxs:
            if op[0] == '': #insertion
                actions[i] = (k, 0, op[1])
            elif op[1] == '': #deletion
                if include_deleted_words:
                    actions[i] = (j, 2, None, op[0])
                else:
                    actions[i] = (j, 2, None)
            else: # substitution
                actions[i] = (j, 1, op[1])
    actions = [actions[i] for i in action_idxs]
    return actions

def get_correct_canvas_positions(alignment):
    correct_positions = []
    idx = 0
    for p in alignment:
        if p[0] == '': continue
        if p[0] == p[1]: correct_positions.append(idx)
        idx += 1
    return correct_positions

def canvas_to_text(canvas, tokenizer, token_type_ids=None):
    if token_type_ids is None:
        token_type_ids = [0] * len(canvas)
    canvas, token_type_ids = clean_canvas(canvas, token_type_ids)
    if len(canvas) == 0: return ''
    else:
        return tokenizer.decode(tokenizer.convert_tokens_to_ids(canvas))

def tag_alignment(alignment, token_type_ids):
    tagged_alignment = []
    j = 0
    for i,tup in enumerate(alignment):
        if tup[0] != '':
            tag = token_type_ids[j]
            j += 1
        else: tag = -1
        tagged_alignment.append([tup[0], tup[1], tag])
    return tagged_alignment

def get_tags_from_alignment(alignment):
    return [tup[2] for tup in alignment if tup[0] != '']

def get_source_canvas(alignment):
    source_tokens = []
    for i,tup in enumerate(alignment):
        if tup[0] != '': source_tokens.append(tup[0])
    return source_tokens

def get_target_canvas(alignment):
    target_tokens = []
    for i,tup in enumerate(alignment):
        if tup[1] != '': target_tokens.append(tup[1])
    return target_tokens

def clean_canvas(canvas, token_type_ids):
    canvas = [c for c,t in zip(canvas, token_type_ids) if t < 3]
    token_type_ids = [t for t in token_type_ids if t < 3]
    # reset the type ids for model insertions
    token_type_ids = [0 if t==1 else t for t in token_type_ids]
    return canvas, token_type_ids

def get_token_type_ids(alignment, idxs):
    token_type_ids = []
    for i,tup in enumerate(alignment):
        if tup[0] != '':
            token_type_ids.append(1 if i in idxs else 0)
    return token_type_ids

class VocabSampler():

    def __init__(self,vocab_weights, vocab_dict, batch_size=1000):
        self.weights = torch.tensor(vocab_weights)
        self.dict = vocab_dict
        self.batch_size=batch_size
        self.samples = torch.multinomial(self.weights, self.batch_size)
        self.sample_idx = 0

    def get(self):
        if self.sample_idx == self.batch_size:
            self.samples = torch.multinomial(self.weights, self.batch_size)
            self.sample_idx = 0
        self.sample_idx += 1
        return self.dict[self.samples[self.sample_idx-1].item()]

####
#### VVVV Most of this now deperacted, moved to models/word_edit_model.py

def noise_alignment_(tagged_alignment, vocab_sampler):
    positions = [i for i,tup in enumerate(tagged_alignment) if tup[-1] < 3] + [-1]
    position = np.random.choice(positions)
    if position >= 0 and tagged_alignment[position][-1] == 0 and \
        tagged_alignment[position][0] == tagged_alignment[position][1]:
        operation = np.random.randint(0, 2)
    else:
        operation = 0

    if operation == 0: # insert
#        token = vocab_dict[torch.multinomial(torch.tensor(vocab_weights), 1).item()]
        token = vocab_sampler.get()
        tagged_alignment = tagged_alignment[:position+1] + [[token, '', 1]] + tagged_alignment[position+1:]
    elif operation == 1: # delete a token
        tup = tagged_alignment[position]
        tagged_alignment = tagged_alignment[:position] + [['', tup[1], -1]] + [[tup[0], '', 3]] + tagged_alignment[position+1:]

    return tagged_alignment


def noise_alignment(tagged_alignment, vocab_weights, vocab_dict, max_noise=3, noise_frac=0.2):
    """
    Noise an alignment for training. This randomly edits the source canvas of the alignment, and updates
    the alignment accordingly
    """
    canv_len = len([tup[0] for tup in tagged_alignment if tup[-1] < 3])
    n_false_edits = int(np.ceil(noise_frac * canv_len))
    for i in range(n_false_edits):
#    while True:
        if random.random() > noise_frac: break
        tagged_alignment = noise_alignment_(tagged_alignment, vocab_weights, vocab_dict)
    return tagged_alignment

def sample_actions(alignment, tokenizer, token_type_ids, vocab_weights,
                  vocab_dict, noise_frac=0.0):
    """
    Sample a set of actions for training. This randomly pushes the alignment forward a few steps
    and then returns the set of remaining actions to turn the source into the target
    """
    if token_type_ids is None:
        token_type_ids = [0 for tup in alignment if tup[0] != '']
    alignment = tag_alignment(alignment, token_type_ids)

    non_const_ops = get_non_const_ops(alignment)
#    print(non_const_ops)
    ops_length = len(non_const_ops) + 1 # +1 operation for stopping


    #sample length
    length = np.random.randint(0,ops_length) # allow sampling any state

    #sample operations
    rand_ops = np.random.permutation(non_const_ops)
    operations = rand_ops[:length]
#    return_ops = rand_ops[length:]

    alignment = operate_tagged_alignment(alignment, operations, agent=0)
    alignment = noise_alignment(alignment, vocab_weights, vocab_dict, noise_frac=noise_frac)
    return_ops = get_non_const_ops(alignment)

    actions = get_actions_from_tagged_alignment(alignment, return_ops)
    if len(actions) == 0: actions = [(1, None, None, None)]
    else: actions = [(0, a[0], a[1], tokenizer.convert_tokens_to_ids(a[2])) for a in actions]

    canvas = get_source_canvas(alignment)
    token_type_ids = get_tags_from_alignment(alignment)

    return canvas, token_type_ids, actions, ops_length

def sample_trajectory(alignment, tokenizer, token_type_ids, vocab_sampler,
                      noise_prob=0.2, max_traj_length=64):
    """
    A more principled approach to noising the canvas. Randomly push the alignment forward
    while randomly making errors (noise).
    """

    alignment = tag_alignment(alignment, token_type_ids)
    ops = get_non_const_ops(alignment)
    ops = np.random.permutation(ops)
    ops_len = len(ops)
    n_errors = 0
    n_ops = 0
    trajectory = []
    trajectory.append((n_errors, n_ops))
    for i in range(max_traj_length):
        if random.random() > noise_prob:
            if n_ops == ops_len and n_errors == 0: break
            if random.random() <= n_errors / (n_errors + ops_len - n_ops):
                n_errors -= 1
            else: n_ops += 1
        else:
            n_errors += 1
        trajectory.append((n_errors, n_ops))
    traj_length = len(trajectory)
    traj_idx = np.random.randint(0,traj_length) # allow sampling any state
    n_errors, n_ops = trajectory[traj_idx]
    alignment = operate_tagged_alignment(alignment, ops[:n_ops])
#    print(alignment)
    for i in range(n_errors):
        alignment = noise_alignment_(alignment, vocab_sampler)
    return_ops = get_non_const_ops(alignment)
    actions = get_actions_from_tagged_alignment(alignment, return_ops)
    if len(actions) == 0: actions = [(1, None, None, None)]
    else: actions = [(0, a[0], a[1], tokenizer.convert_tokens_to_ids(a[2])) for a in actions]

    canvas = get_source_canvas(alignment)
    token_type_ids = get_tags_from_alignment(alignment)

    return canvas, token_type_ids, actions, traj_length


