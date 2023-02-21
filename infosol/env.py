import torch
import copy
import torch.nn.functional as F
import random
import numpy as np
import torch.nn.functional as F

from transformers import BertModel, AutoTokenizer, T5ForConditionalGeneration, BertConfig
from nltk.translate.bleu_score import sentence_bleu

from infosol.alignment import batch_align, align_canvas, score_token_idf, Canvas, batch_align_canvases

class WordEditOracle(torch.nn.Module):
    """
    Oracle Model
    """

    def __init__(self, align_model, align_tokenizer, idf_dict, adjacent_ops=False, sort_ops='sort',
                 baseline_score=0.3, adj_range=1, min_alignment_score=0.7,
                 avoid_delete=True, complete_words=False, token_no_space=None,
                 contiguous_edits=False,
#                 n_contiguous_edits=1
                ):
        """
        :param idf_dict: dict of idf scores for ranking oracle actions
        :param adjacent_ops: whether to limit the model to making adjacent edits
        :param sort_ops: whether to sort the actions returned by the oracle, sorting is done according to idf scores. This is a string (!), can be 'sort', 'random', or 'l2r' (left2right)
        :param n_return_actions: number of actions the oracle returns. 0 means return all actions
        :param baseline_score: baseline score for computing alignment, 0.3 empirically gives "natural" alignments
        :param adj_range: adjacency range for limiting oracle to adjacent edits
        :param avoid_delete: whether to avoid deletion edits
        """
        super().__init__()
        self.idf_dict = idf_dict
#        self.n_contiguous_edits = n_contiguous_edits
        self.contiguous_edits = contiguous_edits
        self.max_idf_score = np.max(list(self.idf_dict.values()))
        self.adjacent_ops = adjacent_ops
        self.adj_range = adj_range
        self.complete_words = complete_words
        self.token_no_space = token_no_space
        if sort_ops not in ('sort', 'random', 'l2r'):
            raise ValueError('unrecognized value for sort_ops: {}'.format(sort_ops))
        self.sort_ops = sort_ops
        self.baseline_score = baseline_score
        self.avoid_delete = avoid_delete
        
        self.align_model = align_model
        self.align_tokenizer = align_tokenizer

    def edit(self, canvas, target, alignment=None, return_alignment=False,
             device=torch.device('cpu'), k=1):
        """
        Make an edit
        """
        if not alignment is None:
            if canvas != alignment.get_source_canvas():
                raise ValueError('canvas does not match alignment')
            if target != alignment.get_target_tokens():
                raise ValueError('target does not match alignment')
        else:
            alignment = align_canvas(canvas, target, self.align_model,
                self.align_tokenizer, baseline_score=self.baseline_score,
                device=device)
        self.edit_alignment(alignment, k=k)
        canvas = alignment.get_source_canvas()

        ret_tup = canvas
        if return_alignment: ret_tup = (canvas, alignment)
        return ret_tup

    def edit_alignment(self, alignment, k=1):
        """
        Make an edit to an alignment
        """
        op_idxs = self.get_action_idxs(alignment)
        op_idxs = self.sort_action_idxs(op_idxs, alignment)
        op_idxs = self.choose_action_idxs(op_idxs, alignment, k)
        op_idxs = list(set(op_idxs))
        alignment.push_forward(op_idxs, agent=1)
        return len(op_idxs) > 0
#
#        if len(op_idxs) > 0:
#            idxs_to_push = [op_idxs[0]]
#            for op in op_idxs[1:]:
#                if len(idxs_to_push) >= self.n_contiguous_edits: break
#                else:
#                    cond1 = op == max(idxs_to_push) + 1
#                    cond2 = op == min(idxs_to_push) - 1
#                    if cond1 or cond2:
#                        idxs_to_push.append(op)
#            if not self.complete_words:
#                alignment.push_forward(idxs_to_push, agent=1)
#            else:
#                idxs_to_push = list(set(idxs_to_push))
#                idxs_to_push_ = []
#                for op in idxs_to_push:
#                    for case, rel_idx in zip(
#                        (alignment.is_insertion, alignment.is_deletion),
#                        (1,0)):
#                        if case(op):
#                            for j in range(op, 0, -1):
#                                if case(j-1)\
#                                    and self.token_no_space(alignment.alignment[j][rel_idx]):
#    #                                and alignment.alignment[j][rel_idx].startswith('#'):
#                                    idxs_to_push_.append(j-1)
#                                else: break
#                            for j in range(op+1, len(alignment)):
#                                if case(j)\
#                                   and self.token_no_space(alignment.alignment[j][rel_idx]):
#    #                               and alignment.alignment[j][rel_idx].startswith('#'):
#                                    idxs_to_push_.append(j)
#                                else: break
#    #                print(idxs_to_push)
#                idxs_to_push = list(set(idxs_to_push + idxs_to_push_))
#                alignment.push_forward(list(idxs_to_push), agent=1)
#            return True
#        else: return False

    def get_action_idxs(self, alignment):
        """
        Choose actions from alignment
        """

        # retrieve idxs in the alignment that correspond to an edit
        op_idxs = alignment.get_non_const_ops()
        if len(op_idxs) == 0: return []

        # if only allowing adjacent edits, limit idxs to adjacent idxs
        if self.adjacent_ops:
            op_idxs = alignment.get_adjacent_ops(op_idxs, adj_range=self.adj_range)
        return op_idxs

    def sort_action_idxs(self, op_idxs, alignment):
        """
        Sort possible actions, based on alignment scores and idf scores
        """
        if self.sort_ops == 'sort':
            # score using idf score and alignment score
            def score_op_idx(idx, alignment, max_idf_score):
                score = 0
                if alignment.is_deletion(idx): #deletions get treated specially
                    if self.avoid_delete:
                        score = 0
                    else: # note that the alignment score for a deletion is the baseline score, usually 0.3
                    # should have a more principled way of scoring deletions?
                        score = alignment.scores[idx] * score_token_idf(
                            alignment.alignment[idx][0], self.idf_dict, max_idf_score)
                else: # For insertions, the alignment score is the baseline score, usually 0.3
                    score = (1-alignment.scores[idx]) * score_token_idf(
                        alignment.alignment[idx][1], self.idf_dict, max_idf_score)
                return score
            idx_scores = [score_op_idx(i, alignment, self.max_idf_score) for i in op_idxs]
            op_idxs = list(np.asarray(op_idxs)[np.argsort(idx_scores)[::-1]])
        elif self.sort_ops == 'random':
            op_idxs = [op_idxs[i] for i in np.random.choice(
                np.arange(len(op_idxs)), len(op_idxs), replace=False)]
        # if going left to right, we still want to prioritize a set of "misaligned" words
        elif self.sort_ops == 'l2r':
            priority_ops = [i for i in op_idxs if alignment_scores[i] < 0.8]
            if len(priority_ops) > 0:
                op_idxs = priority_ops
        return op_idxs

    def choose_action_idxs(self, op_idxs, alignment, k):

        """
        Choose which actions to take based on some heuristics
        """

        def complete_word(op):
            """
            Completes actions so they operate on whole words, not tokens
            """
            complete_idxs = [op]
            for case, rel_idx in zip(
                (alignment.is_addition, alignment.is_deletion),
                (1,0)
            ):
                try:
                    if case(op):
                        for j in range(op, 0, -1): #search backwards for word boundary
                            if case(j-1) and self.token_no_space(alignment.alignment[j][rel_idx]):
                                complete_idxs.append(j-1)
                            else: break
                        for j in range(op+1, len(alignment)): #search forwards for word boundary
                            if case(j) and self.token_no_space(alignment.alignment[j][rel_idx]):
                                complete_idxs.append(j)
                            else: break
                except TypeError as e:
                    print(alignment)
                    raise e
            return complete_idxs

        def add_op(op, l):
            if self.complete_words:
                l.extend(complete_word(op))
            else:
                l.append(op)
            return l

        idxs_to_push = []
        for i,op in enumerate(op_idxs):
            if len(idxs_to_push) >= k: break
            if op in idxs_to_push: continue

            if not self.contiguous_edits:
                idxs_to_push = add_op(op, idxs_to_push)
            else:
                # start building set of contiguous edits
                contiguous_idxs = add_op(op, [])
                for opp in op_idxs[i+1:]:
                    if opp in idxs_to_push: continue
                    if opp in contiguous_idxs: continue
                    if len(idxs_to_push) + len(contiguous_idxs) >= k: break

                    cond1 = opp == max(contiguous_idxs) + 1
                    cond2 = opp == min(contiguous_idxs) - 1
                    if cond1 or cond2:
                        contiguous_idxs = add_op(opp, contiguous_idxs)
                idxs_to_push.extend(contiguous_idxs)

        return idxs_to_push

#        if len(op_idxs) > 0:
#            idxs_to_push = [op_idxs[0]]
#            for op in op_idxs[1:]:
#                if len(idxs_to_push) >= self.n_contiguous_edits: break
#                else:
#                    cond1 = op == max(idxs_to_push) + 1
#                    cond2 = op == min(idxs_to_push) - 1
#                    if cond1 or cond2:
#                        idxs_to_push.append(op)
#            if not self.complete_words:
#                alignment.push_forward(idxs_to_push, agent=1)
#            else:
#                idxs_to_push = list(set(idxs_to_push))
#                idxs_to_push_ = []
#                for op in idxs_to_push:
#                    for case, rel_idx in zip(
#                        (alignment.is_insertion, alignment.is_deletion),
#                        (1,0)):
#                        if case(op):
#                            for j in range(op, 0, -1):
#                                if case(j-1)\
#                                    and self.token_no_space(alignment.alignment[j][rel_idx]):
#    #                                and alignment.alignment[j][rel_idx].startswith('#'):
#                                    idxs_to_push_.append(j-1)
#                                else: break
#                            for j in range(op+1, len(alignment)):
#                                if case(j)\
#                                   and self.token_no_space(alignment.alignment[j][rel_idx]):
#    #                               and alignment.alignment[j][rel_idx].startswith('#'):
#                                    idxs_to_push_.append(j)
#                                else: break
#    #                print(idxs_to_push)
#                idxs_to_push = list(set(idxs_to_push + idxs_to_push_))
#                alignment.push_forward(list(idxs_to_push), agent=1)
# 

class EditingEnvironment():

    def __init__(self, oracle, oracle_stop_p = 0.5, n_oracle_edits = -1, bleu_weights=(0.25,0.25,0.25,0.25)):
        """
        :param oracle_stop_p: the oracle makes edits until it stops with probability oracle_stop_p
        :param bleu_weights: used for the reward, not important since we don't use rewards
        """
        self.oracle = oracle
        self.oracle_stop_p = oracle_stop_p
        self.n_oracle_edits = n_oracle_edits
        self.bleu_weights=bleu_weights

        self.alignment = None

    def check_input_(self, canvas=None, target_tokens=None, alignment=None, device=torch.device('cpu')):
        if all((alignment is None, canvas is None, target_tokens is None)):
            raise ValueError('alignment, canvas and target cannot all be None')
        elif alignment is None and (canvas is None or target_tokens is None):
            raise ValueError('canvas and target cannot both be None')

        if not alignment is None:
            if canvas is None:
                canvas = alignment.get_source_canvas()
            if target_tokens is None:
                target_tokens = alignment.get_target_tokens()

            if canvas != alignment.get_source_canvas():
                raise ValueError('canvas does not match alignment')
            if target_tokens != alignment.get_target_tokens():
                raise ValueError('target does not match alignment')
            alignment = copy.deepcopy(alignment)
        else:
            alignment = align_canvas(canvas, target_tokens, self.oracle.align_model,
                self.oracle.align_tokenizer, baseline_score=self.oracle.baseline_score,
                device=device)
        return canvas, target_tokens, alignment

    def oracle_edit(self, canvas=None, target_tokens=None, alignment=None, device=torch.device('cpu'), return_alignment=False):
        canvas, target_tokens, alignment = self.check_input_(canvas, target_tokens, alignment, device=device)
        """
        Let the oracle make an edit
        """
        if self.n_oracle_edits >= 0:
            self.oracle.edit_alignment(alignment, k=self.n_oracle_edits)
#            for i in range(self.n_oracle_edits):
#                self.oracle.edit_alignment(alignment)
        else:
            k = np.random.geometric(self.oracle_stop_p)
            self.oracle.edit_alignment(alignment, k=k)
#            while True:
#                self.oracle.edit_alignment(alignment)
#                if np.random.random() <= self.oracle_stop_p:
#                    break
        if return_alignment:
            return alignment
        else:
            return alignment.get_source_canvas()

    def reset(self, target_tokens=None, source_canvas=None, alignment=None, device=torch.device('cpu'),
             return_alignment=False):
        """
        Reset the environment. Can either specify the target and source or the alignment (saves the cost of
        aligning the source to the target
        """
        self.alignment = self.oracle_edit(source_canvas, target_tokens, alignment, return_alignment=True)
        if return_alignment:
            return copy.deepcopy(self.alignment)
        else:
            return copy.deepcopy(self.alignment.get_source_canvas())

    def compute_reward(self, prev_canvas, agent_canvas, target_tokens):
        tokenizer = self.oracle.align_tokenizer

        prev_text = prev_canvas.render(tokenizer)
        target_text = Canvas(target_tokens).render(tokenizer)
        agent_text = agent_canvas.render(tokenizer)

        prev_bleu_score = sentence_bleu([target_text],
            prev_text, weights=self.bleu_weights)
        agent_bleu_score = sentence_bleu([target_text],
            agent_text, weights=self.bleu_weights)
        delta = agent_bleu_score - prev_bleu_score
        return delta, agent_bleu_score, prev_bleu_score

    def step(self, canvas, return_alignment=False, device=torch.device('cpu')):
        """
        One step of the environment. Consists of oracle edit, computing rewards etc.
        """
        prev_canvas, target_tokens = self.alignment.get_source_canvas(), self.alignment.get_target_tokens()
        reward, agent_score, _ = self.compute_reward(prev_canvas, canvas, target_tokens)

        self.alignment = self.oracle_edit(canvas=canvas, target_tokens=target_tokens, device=device, return_alignment=True)
        if return_alignment:
            return copy.deepcopy(self.alignment), reward
        else:
            return copy.deepcopy(self.alignment.get_source_canvas()), reward
