import torch
import tqdm
import os
import time
import pdb
import copy
import torch.nn.functional as F
import random
import operator
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp

from queue import PriorityQueue
from infosol.alignment import *
from infosol.decoding import ParallelDecodingMixin
from transformers import BertModel, AutoTokenizer, T5ForConditionalGeneration, BertConfig, AutoConfig, AutoModel
from transformers.models.bart.modeling_bart import BartEncoder, BartModel, BartForConditionalGeneration, BartDecoder
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

MP_TIMEOUT = 2 #2 seconds timeout

def log_factorial(n):
    if n <= 1: return 0.
    return np.log(np.arange(2,n+1)).sum()

class BeamSearchNode():

    """
    Class for beam search
    """

    def __init__(self, prevNode, canvas, stop, logp, length):
        self.prevNonde = prevNode
        self.canvas = canvas
        self.stop = stop
        self.logp = logp
        self.len = length

    def eval(self, len_normalize=False):
        if len_normalize:
            return self.logp + log_factorial(self.len-1)
        else: return self.logp

    def __eq__(self, other):
        return self.eval() == other.eval()

    def __lt__(self, other):
        return self.eval() < other.eval()

    def __gt__(self, other):
        return self.eval() > other.eval()

    def __le__(self, other):
        return self.eval() <= other.eval()

    def __ge__(self, other):
        return self.eval() >= other.eval()

def noise_alignment(alignment, vocab_sampler):
    """
    Make a random edit to the source canvas and update the alignment
    """
    canvas = alignment.get_source_canvas()
    positions = [-1] + [i for i in range(len(canvas)) if canvas.type_ids[i] < 3]
    position = np.random.choice(positions)
    if position >= 0:
        operation = np.random.randint(0, 3)
    else:
        operation = 0

    if operation < 2: # insert and substitute
        token = vocab_sampler.get()
        alignment.operate(position, operation, token)
    elif operation == 2: # delete a token
        alignment.operate(position, operation, None)

def sample_trajectory(alignment, tokenizer, vocab_sampler, noise_prob=0.2,
                      max_traj_length=64):

    """
    Sample a trajectory for training. Take the alignment and randomly push it forward
    a few steps (make the source more like the target). Inject noise by simulating a noisy
    process where you make errors when pushing the alignment forward.
    """

    ops = alignment.get_non_const_ops()
    ops = np.random.permutation(ops)
    ops_len = len(ops)
    n_errors = 0
    n_ops = 0
    trajectory = []
    trajectory.append((n_errors, n_ops))
    for i in range(max_traj_length):
        if random.random() > noise_prob:
            if n_ops == ops_len and n_errors == 0: break
            if random.random() <= n_errors / (n_errors + ops_len - n_ops): # you might fix an error
                n_errors -= 1
            else: n_ops += 1
        else:
            n_errors += 1
        trajectory.append((n_errors, n_ops))
    traj_length = len(trajectory)
    traj_idx = np.random.randint(0,traj_length) # sample state from trajectory
    n_errors, n_ops = trajectory[traj_idx]
    alignment.push_forward(ops[:n_ops])
    for i in range(n_errors):
        noise_alignment(alignment, vocab_sampler)
    return_ops = alignment.get_non_const_ops()
    actions = alignment.get_actions(return_ops)
    if len(actions) == 0: actions = [(1, None, None, None)]
    else: actions = [(0, a[0], a[1], tokenizer.convert_tokens_to_ids(a[2])) for a in actions]

    canvas = alignment.get_source_canvas()

    return canvas, actions, traj_length

class mlp_head(torch.nn.Module):
    """
    MLP head to put on top of BERT
    """
    
    def __init__(self, hidden_size, out_size, layer_norm_eps):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps),
            torch.nn.Linear(hidden_size, out_size, bias=False)
            )
    
    def forward(self, inp):
        out = self.head(inp)
        return out

def draw_multi(p_values):
    return np.where(np.random.multinomial(1, p_values) == 1)[0][0]

class BertEditor(torch.nn.Module, ParallelDecodingMixin): #TODO: clean up, split out Bart model
    """
    Main editor model
    """
    
    def __init__(self, model_type = 'bert', model_file='bert-base-uncased', is_decoder=False,
                type_vocab_size=2, use_type_ids=True, training_noise=0.2,
                tokenizer=None, vocab_sampler=None):
        """
        :param model_file: model file to initialize bert model, default set to pretrained model from huggingface
        :param use_memory: whether to use memory of oracle actions
        :param is_decoder: whether to use decoder architecture for BERT
        :param cat_embeds: whether to concatenate memory to inputs instead of feeding to cross attention layers (in decoder case)
        """
        super().__init__()
        self.model_type = model_type
        self.use_type_ids = use_type_ids
        self.training_noise = training_noise

        self.tokenizer = tokenizer
        vocab_size = len(self.tokenizer)
        self.vocab_sampler = vocab_sampler
        self.delete_type_id = 3

        if self.model_type == 'bert':
            self.base_model = BertModel.from_pretrained(model_file)
            self.base_model.resize_token_embeddings(vocab_size)
            self.base_model.embeddings.token_type_embeddings = self.base_model._get_resized_embeddings(
                self.base_model.embeddings.token_type_embeddings, 5) # resize token type embeddings
            hidden_size = self.base_model.config.hidden_size
            layer_norm_eps = self.base_model.config.layer_norm_eps
        elif self.model_type in ['bart', 'bart-large']:
            config = AutoConfig.from_pretrained(model_file)
            config.n_type_ids = 5
            self.base_model = BartForConditionalGenerationTType.from_pretrained(
                model_file, config=config).model.encoder
            hidden_size = config.hidden_size
            layer_norm_eps = 1e-12

        self.emission_head = mlp_head(hidden_size, 1, layer_norm_eps) # decides when to "emit" a document/stop
        self.location_head = mlp_head(hidden_size, 1, layer_norm_eps) # decides where to edit
        self.operation_head = mlp_head(hidden_size, 3, layer_norm_eps) # decides what operation to do
        self.vocabulary_head = mlp_head(hidden_size, vocab_size * 2, layer_norm_eps) # decides what token to insert/substitute for

    def forward(self, input_ids, attention_mask, encoder_states=None, encoder_mask=None, token_type_ids=None,
                return_hidden_states=False):

        N, L = input_ids.shape

        if encoder_states is None:
            encoder_states = torch.zeros((N, 1, self.base_model.config.hidden_size)).to(input_ids.device)
            encoder_mask = torch.zeros((N, 1), dtype=torch.int32).to(input_ids.device)

        if not self.use_type_ids:
            token_type_ids = None

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(attention_mask)
        token_type_ids = torch.cat([
            torch.zeros_like(encoder_mask),
            token_type_ids],
            dim=1)
        attention_mask = torch.cat(
            [encoder_mask, attention_mask], dim=1)

        # TODO: split into separate model classes
        if self.model_type == 'bert':
            inputs_embeds = self.base_model.embeddings.word_embeddings(input_ids) #embed token ids
            inputs_embeds = torch.cat([encoder_states, inputs_embeds], dim=1) #concatenate with encoder states
            inputs_embeds = self.base_model.embeddings(inputs_embeds=inputs_embeds,
                token_type_ids=token_type_ids)
            out = self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        elif self.model_type == 'bart' or self.model_type == 'bart-large':
            inputs_embeds = self.base_model.embed_tokens(input_ids)
            inputs_embeds = torch.cat([encoder_states, inputs_embeds], dim=1) #concatenate with encoder states
            out = self.base_model(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        prefix_len = encoder_states.shape[1]

        hidden_states = out[0][:, prefix_len:, :]
        emission_out = self.emission_head(hidden_states[:,0,:]).squeeze(-1)
        location_out = self.location_head(hidden_states).squeeze(-1)
        operation_out = self.operation_head(hidden_states)
        vocabulary_out = self.vocabulary_head(hidden_states)
        vocabulary_out = vocabulary_out.reshape(N, L, 2, -1)
        V = vocabulary_out.shape[-1]

#        if np.isnan(emission_out.detach().cpu()).any():
#            pdb.set_trace()

        # Location mask
        # Mask out padding
        padding_mask = 1 - attention_mask[:, prefix_len:]
        # Mask out stricken tokens
        stricken_mask = (token_type_ids[:, prefix_len:] >= self.delete_type_id) # cannot insert, delete or sub stricken-out tokens <-- may want to allow substitutions?
        # Mask out EOS tokens
        eos_mask = torch.zeros_like(attention_mask[:, prefix_len:])
        attention_lengths = attention_mask[:, prefix_len:].sum(dim=1, dtype=torch.int32)
        for i in range(attention_lengths.shape[0]):
            eos_mask[i, attention_lengths[i].item()-1] = 1
        location_mask = torch.clamp(padding_mask + stricken_mask + eos_mask, max=1)
        location_out += location_mask * -1e20

        # Operation mask
        # BOS mask
        bos_mask = torch.zeros_like(operation_out)
        bos_mask[:, 0, 1:] = 1
        operation_mask = torch.clamp(bos_mask, max=1)
        operation_out += operation_mask * -1e20

        # Vocabulary mask
        # Can't sub token for same one
        sub_mask = torch.zeros_like(vocabulary_out)
        sub_mask[:, :, 1, :] = F.one_hot(input_ids, num_classes=V)
        vocabulary_mask = torch.clamp(sub_mask, max=1)
        vocabulary_out += vocabulary_mask * -1e20

        return_out = (emission_out, location_out, operation_out, vocabulary_out)
        if return_hidden_states:
            return_out = return_out + (hidden_states,)
        return return_out

    def compute_instance_loss(self, emission_out, location_out, operation_out, vocabulary_out,
                              emission_idx, location_idxs, operation_idxs, vocabulary_idxs, ops_len):
        """
        Calculate cross entropy loss based on model outputs and gold targets
        """
        n_actions = max(emission_out.shape[0] - 1, 1)

        emission_loss = F.binary_cross_entropy(torch.sigmoid(emission_out), emission_idx.float())
        emission_mask = (1-emission_idx).type(torch.int32)

        location_loss = F.cross_entropy(location_out, location_idxs, reduction='none')
        location_loss = (location_loss * emission_mask).sum()

        # TODO: should technically mask here for bos, only viable operaiton is insertion
        operation_loss = F.cross_entropy(
            operation_out.gather(1,
                location_idxs.reshape(-1,1,1).expand(-1,-1,operation_out.size(-1))).squeeze(1),
            operation_idxs, reduction='none')
        # multiply by a mask so that don't count the loss in cases where the model should stop
        operation_loss = (operation_loss * emission_mask).sum()
       
       # TODO: should also technically mask here for sub identical token, since not allowed
        vocabulary_loss = F.cross_entropy(
            vocabulary_out.gather(1,
                location_idxs.reshape(-1,1,1,1).expand(-1,-1,2,vocabulary_out.size(-1))).squeeze(1).gather(1,
                    torch.clamp(operation_idxs, max=1).reshape(-1,1,1).expand(-1,1,vocabulary_out.size(-1))).squeeze(1),
            vocabulary_idxs, reduction='none') # <- NOT averaged
        # only use vocabulary loss for substitution and insertion opertions (deletion mask)
        deletion_mask = 1 - (operation_idxs == 2).type(torch.long) # <- 2 = deletion operation
        vocabulary_loss = (vocabulary_loss * deletion_mask * emission_mask).sum()

        # scaling
        total_loss = emission_loss + location_loss + operation_loss + vocabulary_loss # this is already averaged by n-t
        total_loss = -log_factorial(ops_len) + ops_len * total_loss / n_actions

        return total_loss, emission_loss, location_loss, operation_loss, vocabulary_loss

    def compute_loss(self, input_ids, attention_mask, actions, prefix_lengths, ops_lengths,
                     encoder_states=None, encoder_mask=None, token_type_ids=None):
        #TODO: get rid of prefix_lenghts (outdated)
        """
        Computes the loss for a batch. Expands the batch when there are multiple actions to compute loss over all actions.
        Thus, for each entry in the batch, creates N entry for each action from the oracle that the model should predict
        """
        emission_out, location_out, operation_out, vocabulary_out = self.forward(input_ids, attention_mask,
            encoder_states=encoder_states, encoder_mask=encoder_mask, token_type_ids=token_type_ids)[:4]

#        if np.isnan(emission_out.detach().cpu()).any():
#            pdb.set_trace()

        batch_size = location_out.size(0)
        batch_loss, emission_loss, location_loss, operation_loss, vocabulary_loss = [
            torch.tensor(0, dtype=torch.double, device=location_out.device)]*5 # create tensors to track loss
        # the number of actions that the oracle returns for each state could differ, so we have to iterate over the batch
        for e_out, l_out, o_out, v_out, (e_idxs, l_idxs, o_idxs, v_idxs), p_len, o_len in zip(
            emission_out, location_out, operation_out, vocabulary_out, actions, prefix_lengths, ops_lengths):
            n_ops = l_idxs.size(0)
            instance_loss, e_loss, l_loss, o_loss, v_loss = self.compute_instance_loss(
                # expand by number of operations that the oracle returned for these inputs
                e_out.unsqueeze(0).expand(n_ops),
                # prefix here is because we are still including a source canvas in the model inputs, which differs for each instance
                l_out[p_len:-1,...].unsqueeze(0).expand(n_ops, -1),
                o_out[p_len:-1,...].unsqueeze(0).expand(n_ops, -1, -1),
                v_out[p_len:-1,...].unsqueeze(0).expand(n_ops, -1, -1, -1),
                e_idxs, l_idxs, o_idxs, v_idxs, o_len
            )

            batch_loss, emission_loss, location_loss, operation_loss, vocabulary_loss = [
                loss + l / batch_size for loss,l in zip(
                    [batch_loss, emission_loss, location_loss, operation_loss, vocabulary_loss],
                    [instance_loss, e_loss, l_loss, o_loss, v_loss])
            ] # aggregate loss

        metrics = {
            'loss': batch_loss.item(),
            'em_loss': emission_loss.item(),
            'loc_loss': location_loss.item(),
            'op_loss': operation_loss.item(),
            'voc_loss': vocabulary_loss.item()
        }

        return batch_loss, metrics

    def prep_canvas(self, canvas, type_ids=None):
        """
        Prepares canvass to feed into the model
        """
        ids = self.tokenizer.convert_tokens_to_ids(canvas.tokens) # convert tokens to ids
        ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
        type_ids = [0] + canvas.type_ids + [0]
        return ids, type_ids, 0

    def prep_canvases(self, canvases, device=torch.device('cpu')):
        """
        Preps canvases into a batch to feed into the model
        """
        canvases, type_ids, prefix_lengths = zip(*[self.prep_canvas(c) for c in canvases])
        max_length = np.max([len(c) for c in canvases])
        input_ids = torch.zeros((len(canvases), max_length), dtype=torch.long, device=device)
        attention_mask = torch.zeros(input_ids.shape, dtype=torch.int32, device=device)
        token_type_ids = torch.zeros_like(attention_mask, device=device)

        for i,(c, tids) in enumerate(zip(canvases, type_ids)):
            input_ids[i, :len(c)] = torch.tensor(c)
            token_type_ids[i, :len(c)] = torch.tensor(tids)
            attention_mask[i, :len(c)] = 1

        return input_ids, token_type_ids, attention_mask, prefix_lengths

    def prep_action(self, action):
        """
        Prep action for model
        """
        emission_idx = action[0]
        if emission_idx == 0:
            location_idx = action[1] + 1 # + 1 for sentinel token
            operation_idx = action[2]
            token_idx = action[3] if action[3] is not None else 0
        else:
            location_idx, operation_idx, token_idx = 0,0,0 # dummy values
        return emission_idx, location_idx, operation_idx, token_idx

    def prep_actions(self, actions, device=torch.device('cpu')): # preps sampled actions for a SINGLE INSTANCE
        """
        Prep set of actions return by the oracle for the model. This is a single state, not for a batch
        """
        return [torch.tensor(a, dtype=torch.long, device=device) for a in zip(
            *[self.prep_action(a) for a in actions])]

    def prep_batch(self, batch, device=torch.device('cpu'), **kwargs):
        """
        Prepares whole batch to feed into the model
        """
        if self.training_noise > 0 and self.vocab_sampler is None:
            raise ValueError('vocab sampler cannot be None when using training noise')
        canvases, actions, ops_lengths = zip(*[sample_trajectory(
            alignment, self.tokenizer, self.vocab_sampler, noise_prob = self.training_noise, **kwargs
        ) for alignment in batch])

        input_ids, token_type_ids, attention_mask, prefix_lengths = self.prep_canvases(
            canvases, device=device)

        actions = [self.prep_actions(a, device) for a in actions]
        # no encoder input in this version
        encoder_states, encoder_mask = None, None

        return input_ids, attention_mask, actions, prefix_lengths, ops_lengths, encoder_states, encoder_mask, token_type_ids

    def move_batch(self, batch, device):
        input_ids, attention_mask, actions, prefix_lengths, ops_lengths, encoder_states, encoder_mask, token_type_ids = batch
        input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
        #encoder_states, encoder_mask = encoder_states.to(device), encoder_mask.to(device)
        actions = [(a[0].to(device), a[1].to(device), a[2].to(device), a[3].to(device)) for a in actions]
        return input_ids, attention_mask, actions, prefix_lengths, ops_lengths, encoder_states, encoder_mask, token_type_ids

    ### Decoding

    def convert_action(self, action):
        """
        Get action from model representation of action
        """
        stop, ref_idx, op_idx, tok_idx = action
        if not ref_idx is None: ref_idx -= 1
        if not tok_idx is None:
            tok_idx = self.tokenizer.convert_ids_to_tokens(int(tok_idx))
        return stop, ref_idx, op_idx, tok_idx

    @staticmethod
    def sample_from_logits(actions, logps):
        action_ps = np.exp(np.asarray(logps))
        action_ps = action_ps / np.sum(action_ps)
        action_idx = draw_multi(action_ps)
        return actions[action_idx], logps[action_idx]

    def forward_canvases(self, canvases, device=torch.device('cpu'), move_to_cpu=False):
        """
        Forward pass taking canvases as inputs
        """
        input_ids, token_type_ids, attention_mask, prefix_lengths = self.prep_canvases(
            canvases, device=device)
        with torch.no_grad():
            emission_out, location_out, operation_out, vocabulary_out = self.forward(
                input_ids, attention_mask, token_type_ids = token_type_ids)[:4]
        if move_to_cpu:
            emission_out, location_out, operation_out, vocabulary_out =\
                    emission_out.cpu(), location_out.cpu(), operation_out.cpu(), vocabulary_out.cpu()
        B = emission_out.shape[0]
        out = []
        for i in range(B):
            plen = prefix_lengths[i]
            e_out, l_out, op_out, v_out = emission_out[i], location_out[i], operation_out[i], vocabulary_out[i]
#            yield e_out, l_out[plen:], op_out[plen:], v_out[plen:]
            out.append((e_out, l_out[plen:], op_out[plen:], v_out[plen:]))
        return out

    def forward_canvas(self, canvas, device=torch.device('cpu')):
#        return list(self.forward_canvases(canvas, device=device))
        return self.forward_canvases(canvas, device=device)

    def get_topk_actions(self, canvas, device=torch.device('cpu'), top_k=1, **kwargs):
        """
        Get top actions from model predictions
        """
        input_ids, token_type_ids, attention_mask, prefix_lengths = self.prep_canvases(
            [canvas], device=device)
        plen = prefix_lengths[0]
        with torch.no_grad():
            emission_out, location_out, operation_out, vocabulary_out = self.forward(
                input_ids, attention_mask, token_type_ids = token_type_ids)[:4]
        actions, logps, stop_lp = get_topk_action_logps(
            emission_out, location_out[:, plen:], operation_out[:, plen:],
            vocabulary_out[:, plen:], top_k=top_k, **kwargs)
        return actions, logps, stop_lp

    @staticmethod
    def ancestral_sample(emission_out, location_out, operation_out, vocabulary_out):
        """
        Ancestral sampling: sample next action based on previous predictions
        """
        stop_p = torch.sigmoid(emission_out)
        if np.random.random() <= stop_p:
            return (1, None, None, None)

        def sample_from_out(out):
            prob = torch.softmax(out, dim=-1)
            return torch.multinomial(prob, 1).item()

        location = sample_from_out(location_out)
        op = sample_from_out(operation_out[location])
        if op == 2: return (0, location, op, None)
        token = sample_from_out(vocabulary_out[location,op])
        return (0, location, op, token)

    @staticmethod
    def get_topk_action_logps(stop_out, location_out, action_out, vocab_out, top_k=10, exclude_stop=False): # warning: this will modify the outputs from the model!
        """
        Searches the model's output distribution for the top k actions. The actions the model can take
        are organized in a tree. The first level corresponds to deciding whether or not to edit, then where to
        edit, then what operation to take, and finally what token to use for insertions/substitutions.
        We search the tree mainting a list of the most likely actions, and since the probability of a node is always 
        lower than that of its parent (since the probabilities multiply and are all leq 1), we can easily determine
        which nodes to explore by exploring them in sorted order.
        """
        class ActionBuffer:

            def __init__(self, top_k):
                self.top_k = top_k
                self.actions = [None] * top_k
                self.logps = [-1e20] * top_k

            def insert(self, action, logp):
                # find insertion index
                for i in range(self.top_k-1, -2, -1):
                    if i == -1:
                        break
                    if logp < self.logps[i]:
                        break

                self.logps = self.logps[:i+1] + [logp] + self.logps[i+1:]
                self.actions = self.actions[:i+1] + [action] + self.actions[i+1:]

                self.logps, self.actions = self.logps[:self.top_k], self.actions[:self.top_k]

            def min_lp(self):
                return self.logps[-1]

#        def iter_torch_sort(sort_obj):
#            """
#            Helper function for iterating over tensors. Much more efficient than directly iterating over tensor
#            since we often don't iterate over all the values
#            """
#            values, indices = sort_obj.values, sort_obj.indices
#        #    values = values.cpu()
#            for i in range(values.shape[0]):
#                yield values[i], indices[i]

        def iter_np_sort(sort_obj, descending=True):
            sorted_indices = np.argsort(sort_obj)
            if descending: sorted_indices = reversed(sorted_indices)
            for i in sorted_indices: #desce
                yield sort_obj[i], i

        # always include first index in frozen_idxs since cannot sub or del the CLS token
        # buffer for actions and their logps
        action_buffer = ActionBuffer(top_k)

        levels = (stop_out, location_out, action_out, vocab_out)
        level_logps = np.array([0,0,0,0], dtype=np.float32)
        # stopping probabilities
        stop_p = torch.sigmoid(stop_out)
        stop_lp = torch.log(stop_p)
        if not exclude_stop:
            action_buffer.insert((1, None, None, None), stop_lp.item())
        level_logps[0] = np.log(1-stop_p)

        def search_level(level, idxs):
            if level > 3: return
            out = levels[level].unsqueeze(0)[idxs]
            logprobs = F.log_softmax(out, dim=-1)
            for lprob, i in iter_np_sort(logprobs.numpy()):
#            for lprob, i in iter_torch_sort(torch.sort(logprobs, descending=True)):
                level_logps[level] = lprob
                cumlp = level_logps[:level+1].sum()
                if cumlp <= action_buffer.min_lp():
                    return
                if level == 3:
                    action_buffer.insert(idxs + (i,), cumlp)
                elif level == 2 and i==2: #deletion action is special because it doesn't specify a token
                    action_buffer.insert(idxs + (i, None,), cumlp)
                else:
                    search_level(level+1, idxs + (i,))

        search_level(1, (0,))

        return action_buffer.actions, action_buffer.logps, stop_lp.item()

    @staticmethod
    def sample(model_out, top_k=None):
        if top_k is None:
            action = BertEditor.ancestral_sample(*model_out)
        else:
            actions, logps, stop_lp = BertEditor.get_topk_action_logps(*model_out, top_k=top_k)
            action, _ = BertEditor.sample_from_logits(actions, logps)
        return action

    def edit(self, canvas, top_k=None, device=torch.device('cpu')):
        """
        Make an edit to a canvas
        """
        model_out = next(self.forward_canvas_([canvas], device=device, move_to_cpu=True))
        action = BertEditor.sample(model_out, top_k=top_k)
        action = self.convert_action(action)
        if action[0]: return True
        canvas.operate(*action[1:], agent=0)
        return False

#    def decode_loop(self, canvases, decode_func, state=None, max_batch_tokens=2048, parallel=False,
#                    n_processes=None, device=torch.device('cpu'), queue_size=2000, return_idx=False):
#        ''' parallelized decoding loop '''
#
#        def enqueue(canvas, state, iter_idx):
#            input_queue.put((canvas, iter_idx, state))
#
#        input_iter = iter(canvases)
#        start_state = state
#        def add_input(idx):
#            # add input to queue
#            c = next(input_iter, None)
#            if c is None:
#                return True
#            else:
#                enqueue(c.copy(), start_state, idx)
#                return False
#
#        def inputs_generator(batch):
#            canvases = [b[0] for b in batch]
#            for i, model_out in enumerate(self.forward_canvas_(canvases, device=device, move_to_cpu=True)):
#                canvas, idx, state = batch[i]
#                yield model_out, canvas, state, idx
#
#        iter_idx = 0
#        finished = False
#        input_queue = PriorityQueue(maxsize=queue_size) # priority queue groups similar sized inputs together
#
#        # fill up queue
#        while not input_queue.full():
#            finished = add_input(iter_idx)
#            iter_idx += 1
#            if finished: break
#
#        if parallel:
#            if n_processes is None: n_processes = os.cpu_count()
#            else: n_processes = min(os.cpu_count(), n_processes)
##            print(f'Using {n_processes} processes')
#            worker_pool = mp.Pool(processes=n_processes)
#
#        start_time = time.time()
#
#        try:
#            while not input_queue.empty():
#                # prepare batch
#                batch = []
#                batch_len = 0
#                batch_n_tokens = 0
#                while True:
#                    if input_queue.empty(): break
#                    # get input
#                    inp = input_queue.get()
#                    canvas, idx, state = inp
#                    n_tokens = len(canvas)
#                    # check if it can fit in batch
#                    new_batch_len = batch_len + 1
#                    new_batch_n_tokens = max(batch_n_tokens, n_tokens) # max because inputs will get padded
#                    # if it doesn't fit, requeue and break
#                    if new_batch_len * new_batch_n_tokens > max_batch_tokens:
#                        enqueue(canvas, state, idx) # requeue
#                        break
#                    # otherwise add to batch
#                    else:
#                        batch.append(inp)
#                        batch_len = new_batch_len
#                        batch_n_tokens = new_batch_n_tokens
#
#                batch_size = len(batch)
#                if batch_size == 0:
#                    raise RuntimeError('Unable to fit any inputs into batch. Try increasing max batch tokens')
#                inputs_feed = list(inputs_generator(batch))
#                if parallel:
#                    chunk_size = batch_size // n_processes
##                    outputs_generator = worker_pool.imap_unordered(decode_func, inputs_feed, chunk_size)
#                    outputs_generator = [worker_pool.apply_async(decode_func, args=(inp,)) for inp in inputs_feed]
#                else:
#                    outputs_generator = map(decode_func, inputs_feed)
#                did_timeout = False
#                for out in outputs_generator:
#                    if parallel:
#                        try:
#                            out = out.get(MP_TIMEOUT)
#                        except mp.TimeoutError:
#                            did_timeout = True
#                            if return_idx:
#                                yield None, None
#                            else:
#                                yield None
#                            continue
#                    canvas, action, stop, state, idx = out
#                    if stop:
#                        if return_idx:
#                            yield canvas, idx
#                        else:
#                            yield canvas
#                        if not finished:
#                            finished = add_input(iter_idx)
#                            iter_idx += 1
#                    else:
#                        action = self.convert_action(action)
#                        canvas.operate(*action[1:], agent=0)
#                        enqueue(canvas.copy(), state, idx)
#                if did_timeout and parallel: #reset the worker pool to avoid accumulating dead processes
#                    print('restarting pool')
#                    worker_pool.terminate()
#                    worker_pool = mp.Pool(processes=n_processes)
##            assert finished
#        finally:
#            if parallel:
#                worker_pool.terminate()
#
#        delta = time.time() - start_time
#        delta = int(1000*delta)
##        print(f'Took {delta} ms')
##        return out_canvases
#        return

    @staticmethod
    def canvas_len(canvas):
        return len(canvas)

    def decode_(self, model_out, canvas, state):
        """
        Regular decoding with sampling
        """
#        if np.random.random() <= 0.01:
#            time.sleep(9999)
#        model_out, canvas, state, idx = inputs
        top_k, max_iter, cur_iter = state
        state = top_k, max_iter, cur_iter + 1
        action = BertEditor.sample(model_out, top_k=top_k)
        if action[0] or cur_iter > max_iter:
#            return canvas, action, True, state, idx
            return canvas, state, True
        else:
            action = self.convert_action(action)
            canvas.operate(*action[1:], agent=0)
#            return canvas, action, False, state, idx
            return canvas, state, False

    def batch_decode(self, canvases, top_k=10, max_iter=32, **kwargs):
        state = top_k, max_iter, 0
        for out in self.decode_loop(canvases, self.decode_, state=state, **kwargs):
            yield out

    def decode(self, canvas, **kwargs):
        return next(self.batch_decode([canvas], **kwargs))

    def depth_decode_(self, model_out, canvas, state):
        """
        Depth decoding: ignores stop actions, continus until reaching a stopping condition. Then
        returns whichever canvas had the highest stopping probability
        """
#        model_out, canvas, state, idx = inputs
        top_k, stop_threshold, max_iter, cur_iter, top_canvas, top_lp = state

        actions, logps, stop_lp = BertEditor.get_topk_action_logps(*model_out, top_k=top_k, exclude_stop=True)
        if stop_lp > top_lp:
            top_lp = stop_lp
            top_canvas = canvas.copy()
        state = (top_k, stop_threshold, max_iter, cur_iter + 1, top_canvas, top_lp)
        if np.exp(stop_lp) >= stop_threshold or cur_iter > max_iter:
            return top_canvas, state, True
#            return top_canvas, None, True, state, idx
        else:
            action, _ = BertEditor.sample_from_logits(actions, logps)
            action = self.convert_action(action)
            canvas.operate(*action[1:], agent=0)
            return canvas, state, False
#            return canvas, action, False, state, idx

    def batch_depth_decode(self, canvases, top_k=10, stop_threshold=0.95, max_iter=64, **kwargs):
        state = top_k, stop_threshold, max_iter, 0, None, -1e20
        for out in self.decode_loop(canvases, self.depth_decode_, state=state, **kwargs):
            yield out

    def depth_decode(self, canvas, **kwargs):
        return next(self.batch_depth_decode([canvas], **kwargs))
#
#    def depth_decode(self, canvas, top_k=20, device=torch.device('cpu'), max_iter=64, stop_threshold=0.95, len_normalize=False, only_stop_lp=False):
#        canvases = PriorityQueue()
#        running_logp = 0
#        canvas = canvas.copy()
#        for i in range(max_iter):
#            actions, logps, stop_lp = self.get_topk_actions(canvas, device, top_k=top_k, exclude_stop=True)
#
#            # add completed canvas
#            if only_stop_lp:
#                canvases.put((-stop_lp, canvas))
#            else:
#                canvases.put((-(running_logp + stop_lp), canvas))
#
#            #stopping condition
#            if np.exp(stop_lp) >= stop_threshold: break
#
#            # make new canvas
#            canvas = canvas.copy()
#            action, lp = self.sample_action_(actions, logps)
#            action = self.convert_action(action)
#            canvas.operate(*action[1:], agent=0)
#            running_logp += lp
#            if len_normalize: running_logp += np.log(i+1)
#        return canvases.get()[1]

    def beam_decode(self, canvas, top_k=1, beam_width=10, max_length=32, len_normalize=False, device=torch.device('cpu')):
        """
        Beam decoding, not actually used
        """

        if top_k > beam_width:
            raise ValueError('top_k cannot be greater than beam_width')

        decoded_batch = []
        endnodes = PriorityQueue()

        node = BeamSearchNode(None, canvas, False, 0, 0)
        nodes = PriorityQueue()

        nodes.put((-node.eval(len_normalize=len_normalize), node))
        qsize = 1

        for i in range(max_length):
            if nodes.empty(): break
            nextnodes = PriorityQueue()
            # go through nodes at current depth
            assert nodes.qsize() <= beam_width, nodes.qsize()
            while not nodes.empty():
                score, n = nodes.get()
                canvas = n.canvas

                if n.stop:
                    endnodes.put((score, n))
                    continue

                actions, logps, _ = self.get_topk_actions(canvas, device, top_k=beam_width)
                for a,lp in zip(actions, logps):
                    edited_canvas = canvas.copy()
                    a = self.convert_action(a)
                    if not a[0]: edited_canvas.operate(*a[1:], agent=0)


                    node = BeamSearchNode(n, edited_canvas, a[0], n.logp + lp, n.len + 1)
                    score = -node.eval(len_normalize=len_normalize)
                    nextnodes.put((score, node))

            for _ in range(beam_width):
                if nextnodes.empty(): break
                score, nn = nextnodes.get()
                nodes.put((score, nn))

        if endnodes.empty():
            for _ in range(top_k):
                score,n = nodes.get()
                endnodes.put((score, n))

        for _ in range(top_k):
            if endnodes.empty(): break
            score, n = endnodes.get()
            decoded_batch.append(n.canvas)

        return decoded_batch

class BartEncoderTType(BartEncoder):

    """
    Token type ids for BART (not supported natively)
    """

    def __init__(self, config, embed_tokens):
        super().__init__(config, embed_tokens)

        self.token_type_embeddings = torch.nn.Embedding(config.n_type_ids, config.d_model)

    def forward(self, input_ids=None, inputs_embeds=None, token_type_ids=None, **kwargs):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        token_type_embeds = self.token_type_embeddings(token_type_ids)
        inputs_embeds += token_type_embeds

        return super().forward(inputs_embeds=inputs_embeds, **kwargs)

class BartModelTType(BartModel):

    def __init__(self, config):
        super(BartModel, self).__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = torch.nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoderTType(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                token_type_ids=None, encoder_outputs = None, **kwargs):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                token_type_ids=token_type_ids
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        return super().forward(input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask,
                               inputs_embeds=inputs_embeds, output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states, return_dict=return_dict,
                               encoder_outputs=encoder_outputs, **kwargs)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class BartForConditionalGenerationTType(BartForConditionalGeneration):

    """
    Derived class to add in the token type ids
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = BartModelTType(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = torch.nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        token_type_ids=None,
        **kwargs # why not absorb all these arguments in kwargs??
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_type_ids=token_type_ids,
            **kwargs
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

class BartS2SEditor(torch.nn.Module, ParallelDecodingMixin):
    """
    S2S model
    """

    def __init__(self, align_model, tokenizer, model_file='facebook/bart-base', from_pretrained=True,
                alignment_baseline_score=0.2):
        super().__init__()

        self.align_model = align_model
        for p in self.align_model.parameters():
            p.requires_grad = False
        self.alignment_baseline_score=alignment_baseline_score
        self.tokenizer = tokenizer
        vocab_size = len(self.tokenizer)

        config = AutoConfig.from_pretrained(model_file)
        config.n_type_ids = 5

        if from_pretrained:
            self.bart_model = BartForConditionalGenerationTType.from_pretrained(
                model_file, config=config)
        else:
            self.bart_model = BartForConditionalGenerationTType.from_config(config)
        self.bart_model.resize_token_embeddings(vocab_size)

    def forward(self, **kwargs):
        return self.bart_model(**kwargs)

    def compute_loss(self, input_ids, token_type_ids, attention_mask, labels):
        model_loss = self.bart_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                    token_type_ids=token_type_ids).loss
        metrics = {'loss': model_loss.item()}
        return model_loss, metrics

    def prep_canvas(self, canvas):
        tokens, type_ids = canvas.tokens, canvas.type_ids
        token_ids = [self.tokenizer.bos_token_id] + \
                self.tokenizer.convert_tokens_to_ids(tokens) +\
                [self.tokenizer.eos_token_id]
        type_ids = [0] + type_ids + [0]
        assert len(token_ids) == len(type_ids)
        return token_ids, type_ids

    def prep_canvases(self, canvases, device=torch.device('cpu')):
        token_ids, type_ids = zip(*[self.prep_canvas(c) for c in canvases])
        max_length = np.max([len(ids) for ids in token_ids])
        input_ids = torch.ones((len(canvases), max_length), dtype=torch.long,
                               device=device) * self.tokenizer.pad_token_id
        attention_mask = torch.zeros(input_ids.shape, dtype=torch.int32, device=device)
        token_type_ids = torch.zeros_like(attention_mask)
        for i in range(len(canvases)):
            input_ids[i, :len(token_ids[i])] = torch.tensor(token_ids[i])
            token_type_ids[i, :len(token_ids[i])] = torch.tensor(type_ids[i])
            attention_mask[i, :len(token_ids[i])] = 1
        return input_ids, attention_mask, token_type_ids

    def prep_target(self, target):
        target = [self.tokenizer.bos_token_id] +\
                self.tokenizer.convert_tokens_to_ids(target)+\
                [self.tokenizer.eos_token_id]
        return target

    def prep_targets(self, targets, device=torch.device('cpu')):
        token_ids = [self.prep_target(t) for t in targets]
        max_length = np.max([len(ids) for ids in token_ids])
        input_ids = torch.ones((len(targets), max_length), dtype=torch.long,
                               device=device) * -100
        for i in range(len(targets)):
            input_ids[i, :len(token_ids[i])] = torch.tensor(token_ids[i])
        return input_ids

    def prep_gen(self, gen):
        return self.tokenizer.convert_tokens_to_ids(gen)

    def prep_generations(self, generations, device=torch.device('cpu')):
        token_ids = [self.prep_gen(g) for g in generations]
        max_length = np.max([len(ids) for ids in token_idxs])
        input_ids = torch.ones((len(targets), max_length), dtype=torch.long,
                               device=device) * self.tokenizer.pad_token_id
        input_lengths = []
        for i in range(len(generations)):
            input_ids[i, :len(token_ids[i])] = torch.tensor(token_ids[i])
            input_lengths.append(len(token_ids[i]))
        return input_ids, input_lengths

    def prep_batch(self, batch, device=torch.device('cpu'), **kwargs):
        canvases = [a.get_source_canvas() for a in batch]
        input_ids, attention_mask, type_ids = self.prep_canvases(canvases, device=device)
        targets = [a.get_target_tokens() for a in batch]
        labels = self.prep_targets(targets, device=device)

        return input_ids, type_ids, attention_mask, labels

    def move_batch(self, batch, device):
        input_ids, type_ids, attention_mask, labels = batch
        return input_ids.to(device), type_ids.to(device), attention_mask.to(device), labels.to(device)

    # ====== DECODING ============

#    @staticmethod
#    def canvas_len(canvas):
#        canvas, gen = canvas
#        return len(canvas)
#
#    def forward_canvases(self, canvases, device=torch.device('cpu'), move_to_cpu=True):
#        batch_size = len(canvases)
#        canvases, gen = zip(*canvases)
#        input_ids, attention_mask, type_ids = self.prep_canvases(canvases, device=device)
#        gen_ids, gen_lengths = self.prep_generations(gen, device=device)
#        with torch.no_grad():
#            model_out = self.bart_model(input_ids=input_ids, attention_mask=attention_mask,
#                                        token_type_ids=type_ids, decoder_input_ids=gen_ids)
#        out = []
#        for i in range(batch_size):


    def batch_generate(self, canvases, device=torch.device('cpu'), **kwargs):
        generations = []
        for c in tqdm.tqdm(canvases):
            input_ids, attention_mask, type_ids = self.prep_canvases([c], device=device)
#            print(input_ids.shape, attention_mask.shape)
            gen = self.bart_model.generate(input_ids=input_ids,
                attention_mask=attention_mask, token_type_ids=type_ids, early_stopping=False,**kwargs)
#            print(gen.shape)
#            for g in gen.cpu().numpy():
#                print(self.tokenizer.decode(g))
            tokens = self.tokenizer.convert_ids_to_tokens(gen[0].cpu().numpy())
            tokens = [t for t in tokens if not t in ('<s>', '</s>', '<pad>')] # get rid of extraneous tokens
#            generations.append(self.tokenizer.convert_ids_to_tokens(gen[0][2:-1].cpu().numpy()))
            generations.append(tokens)
        generated_canvases = []
#        print(generations)
        for alignment in batch_align_canvases(canvases, generations, self.align_model, self.tokenizer,
            baseline_score=self.alignment_baseline_score, device=device):
            alignment.push_forward(alignment.get_non_const_ops())
            generated_canvases.append(alignment.get_source_canvas())
        return generated_canvases

    def generate(self, canvas, **kwargs):
        return self.batch_generate([canvas], **kwargs)[0]

#def sample_action(stop_out, location_out, action_out, vocab_out, input_ids, sample=False):
#    if sample:
#        stop = int(np.random.random() <= stop_out.item())
#    else:
#        stop = int(stop_out.item() > 0.5)
#    if stop == 1:
#        return stop, None, None, None
#
#    if sample:
#        refinement_idx = torch.multinomial(F.softmax(location_out, dim=-1), 1).item()
#    else:
#        refinement_idx = torch.argmax(F.softmax(location_out, dim=-1)).item()
#    if refinement_idx > 0:
#        if sample:
#            action_idx = torch.multinomial(F.softmax(action_out[:, refinement_idx, :], dim=-1), 1).item()
#        else:
#            action_idx = torch.argmax(F.softmax(action_out[:, refinement_idx, :], dim=-1)).item()
#    else: # can only insert when selecting the sentinel
#        action_idx = 0
#
#    if action_idx != 2:
#        if sample:
#            token = torch.multinomial(F.softmax(vocab_out[:, refinement_idx, action_idx, :], dim=-1), 1).squeeze().item()
#        else:
#            top_tokens = torch.argsort(F.softmax(vocab_out[:, refinement_idx, action_idx, :], dim=-1)).squeeze()
#            token = top_tokens[-1].item()
#            if token == input_ids[:, refinement_idx] and action_idx == 1: token = top_tokens[-2].item()
#    else: token = None
#
#    return stop, refinement_idx, action_idx, token
#


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


