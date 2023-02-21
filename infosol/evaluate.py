import argparse
import pickle
import itertools
import json
import torch
import tqdm

from distutils.util import strtobool
from nltk.translate.bleu_score import sentence_bleu

from transformers import AutoTokenizer, BartModel, BertModel
from datasets import load_from_disk

from infosol.alignment import Alignment, Canvas, batch_align_canvases
from infosol.models.word_edit_model import BertEditor, BartS2SEditor
from infosol.env import WordEditOracle, EditingEnvironment

class EvaluationInstance():

    def __init__(self, datum):
        self.done = False
        self.rewards = []
        self.history = []
        self.actions = []
        self.consistency = []
        self.bleu_scores = []
        self.canvas_history = []
        self.canvas_actions = []
        self.target_tokens = datum['target_tokens']
        self.canvas = Canvas(datum['source_tokens'], datum['token_type_ids'])
        self.source_text = datum['source_text']
        self.target_text = datum['target_text']

    def to_dict(self):
        return {
            'source': self.source_text,
            'target': self.target_text,
            'history': self.history,
            'rewards': self.rewards,
            'actions': self.actions,
            'consistency': self.consistency,
            'bleu_scores': self.bleu_scores,
            'canvas history': self.canvas_history,
            'canvas actions': self.canvas_actions
        }


class Evaluate():

    """
    Evaluation job
    """

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--model_path', type=str,
                           default='') #/path/to/WEIGHTS.bin)
        parser.add_argument('--out_path', type=str,
                           default='') #/path/to/save_file.pickle)
        parser.add_argument('--cuda_device', type=int, default=0)
        parser.add_argument('--data_path', type=str, default='') #/path/to/data/)
        parser.add_argument('--max_data', type=int, default=1000)
        parser.add_argument('--idf_path', type=str, default='') #/path/to/idf.pickle)
        parser.add_argument('--n_oracle_edits', type=int, default=3)
        parser.add_argument('--oracle_stop_p', type=float, default=0.5)
        parser.add_argument('--n_episodes', type=int, default=4)
        parser.add_argument('--BLEU_threshold', type=float, default=1.0)
        parser.add_argument('--adjacent_ops', type=lambda x:bool(strtobool(x)), default=True)
        parser.add_argument('--complete_words', type=lambda x:bool(strtobool(x)), default=True)
        parser.add_argument('--contiguous_edits', type=lambda x:bool(strtobool(x)), default=True)
        parser.add_argument('--baseline_alignment_score', type=float, default=0.3)
        parser.add_argument('--sort_ops', type=str, default='sort')
        parser.add_argument('--bleu_ngrams', type=int, default=1)
        parser.add_argument('--keyword_gen', type=lambda x:bool(strtobool(x)), default=False)

    def setup(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.cuda_device))
        print('Loading Environment')
        self.tokenizer, self.align_model, self.oracle, self.env = self.setup_env(args)
        print('Loading Model')
        self.model = self.setup_model(args)

    def setup_model(self, args):
        raise NotImplementedError

    def setup_env(self, args):
        raise NotImplementedError

    def create_instance(self, datum):
        inst = EvaluationInstance(datum)
        alignment = Alignment(alignment=datum['alignment'], scores=datum['alignment_scores'])
        inst.canvas = self.env.reset(alignment=alignment)
        return inst

    def step_instance(self, inst, gen, alignment):
        """
        Advance instance by one steps. This includes letting the oracle make changes, and tracking metrics etc.
        """
        prev_canvas, target_tokens = inst.canvas, inst.target_tokens
        agent_canvas = gen
#        agent_canvas, target_tokens_ = alignment.get_source_canvas(), alignment.get_target_tokens()
#        assert target_tokens == target_tokens_, 'mismatching target tokens {target_tokens} vs. {target_tokens_}'

        reward, agent_score, prev_score = self.env.compute_reward(prev_canvas, agent_canvas, target_tokens)

        if not self.args.keyword_gen:
            inst.canvas = self.env.oracle_edit(alignment=alignment)
        else:
            oracle_canvas = self.env.oracle_edit(alignment=alignment)
            tokens_, type_ids_ = oracle_canvas.tokens, oracle_canvas.type_ids
            keywords = [t for t,tid in zip(tokens_, type_ids_) if tid == 2]
            kw_type_ids = [2] * len(keywords)
            inst.canvas = Canvas(keywords, type_ids=kw_type_ids)

        tokenizer = self.env.oracle.align_tokenizer
        inst.history.append(prev_canvas.render(tokenizer))
        inst.canvas_history.append(prev_canvas.copy())
        inst.bleu_scores.append((prev_score, agent_score))
        inst.actions.append(agent_canvas.render(tokenizer))
        inst.canvas_actions.append(agent_canvas.copy())
        inst.rewards.append(reward)

        prev_text = prev_canvas.render(tokenizer)
        agent_text = agent_canvas.render(tokenizer)
        consistency = sentence_bleu([agent_text], prev_text, weights=(1,0,0,0))
        inst.consistency.append(consistency)

        if len(inst.history) >= self.args.n_episodes:
            inst.done = True
        if max(inst.bleu_scores[-1]) >= self.args.BLEU_threshold:
            inst.done = True

        return inst

    def episode(self, instances):
        """
        One editing episode. Model makes changes, align the model outputs to the targets, let the oracle make changes.
        Note the oracle makes changes when initializing the instances, that's why the order here is model then oracle
        instead of oracle then model as described in the paper
        """
        canvases = [inst.canvas for inst in instances]
        generations = self.gen_model(canvases)
        clean_generations = [g.clean() for g in generations]
        align_model, align_tokenizer = self.env.oracle.align_model, self.env.oracle.align_tokenizer
        targets = [inst.target_tokens for inst in instances]
        alignments = batch_align_canvases(clean_generations, targets, align_model, align_tokenizer, device=self.device)
#        return [self.step_instance(inst, gen, a) for inst,gen,a in zip(instances, generations, alignments)]
        instances = [self.step_instance(inst, gen, a) for inst,gen,a in zip(instances, generations, alignments)]
        finished_instances = [inst for inst in instances if inst.done]
        instances = [inst for inst in instances if not inst.done]
        return instances, finished_instances

    def evaluate_instance(self, datum):
        rewards = []
        history = []
        actions = []
        consistency = []
        bleu_scores = []
        canvas_history = []
        canvas_actions = []
        canvas = self.env.reset(alignment=Alignment(
            alignment=datum['alignment'],
            scores=datum['alignment_scores']))
        for e in range(self.args.n_episodes):
            input_text = canvas.render(self.tokenizer)
            target_text = self.env.alignment.get_target_canvas().render(self.tokenizer)
            history.append(input_text)
            canvas_history.append(canvas.copy())
            inp_score = sentence_bleu([target_text], input_text, weights=self.env.bleu_weights)

            canvas = self.gen_model(canvas)

            output_text = canvas.render(self.tokenizer)
            out_score = sentence_bleu([target_text], output_text, weights=self.env.bleu_weights)
            bleu_scores.append((inp_score, out_score))
            actions.append(output_text)
            canvas_actions.append(canvas.copy())

            canvas = canvas.clean()
            canvas, r = self.env.step(canvas, device=self.device)
            rewards.append(r)
            consistency.append(sentence_bleu([output_text], input_text, weights=(1,0,0,0)))
        return {
            'source': datum['source_text'],
            'target': datum['target_text'],
#            'target': self.env.target_text,
            'history': history,
            'rewards': rewards,
            'actions': actions,
            'consistency': consistency,
            'bleu_scores': bleu_scores,
            'canvas history': canvas_history,
            'canvas actions': canvas_actions
        }

    def gen_model(self, canvases):
        raise NotImplementedError

#    def gen_user(self, prev_canvases, canvases, targets, device=torch.device('cpu')):
#        batch_size = 64
#        user_canvases, rewards = [], []
#        def batch(lst, n=1):
#            l = len(lst)
#            for ndx in range(0, l, n):
#                yield lst[ndx:min(ndx + n, l)]
#        for pc, c, t in tqdm.tqdm(zip(
#            batch(prev_canvases, batch_size), batch(canvases, batch_size),
#            batch(targets, batch_size))):
#            _, user_c, r = self.env.step_(pc, c, t, device)
#            user_canvases.extend(user_c)
#            rewards.extend(r)
#        return zip(user_canvases, rewards)

    def run(self):
        self.model = self.model.eval().to(self.device)
        self.oracle = self.oracle.to(self.device)

        data = load_from_disk(self.args.data_path)['test']
        instances = [self.create_instance(datum) for datum in itertools.islice(data, 0, self.args.max_data)]
        finished_instances = []
        while len(instances) > 0:
            instances, finst_ = self.episode(instances)
            finished_instances.extend(finst_)
#        for e in range(self.args.n_episodes):
#            instances = self.episode(instances)
#        results = [inst.to_dict() for inst in instances]
        results = [inst.to_dict() for inst in finished_instances]
#        for datum in tqdm.tqdm(itertools.islice(data,0,self.args.max_data),
#                               total=self.args.max_data):
#            res = self.evaluate_instance(datum)
#            results.append(res)
        with open(self.args.out_path, 'wb') as f:
            pickle.dump(results, f)

class EvaluateEditor(Evaluate):

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--top_k', type=int, default=10)
        parser.add_argument('--stop_threshold', type=float, default=0.95)

    def gen_model(self, canvases):
        canvases = list(tqdm.tqdm(self.model.batch_depth_decode(
            canvases,
            device=self.device,
            top_k=self.args.top_k,
            stop_threshold=self.args.stop_threshold,
            return_idx=True
        ), total=len(canvases)))
        canvases = [(i,c) for c,i in canvases]
        canvases = [c for i,c in sorted(canvases)]
        return canvases

class EvaluateS2S(Evaluate):

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--do_sample', type=lambda x:bool(strtobool(x)), default=False)
        parser.add_argument('--top_p', type=float, default=0.95)
        parser.add_argument('--max_length', type=int, default=64)
        parser.add_argument('--num_beams', type=int, default=10)
        parser.add_argument('--length_penalty', type=float, default=1.0)

    def gen_model(self, canvases):
        return self.model.batch_generate(canvases, device=self.device,
            do_sample=self.args.do_sample, top_p=self.args.top_p, max_length=self.args.max_length,
            num_beams=self.args.num_beams, length_penalty=self.args.length_penalty)

class EvaluateBart(Evaluate):

    def setup_env(self, args):
        tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
        align_model = BartModel.from_pretrained('facebook/bart-base').encoder
        with open(self.args.idf_path, 'rb') as f:
            idf_dict = pickle.load(f)
        oracle = WordEditOracle(align_model, tokenizer, idf_dict=idf_dict,
                                sort_ops='sort', adjacent_ops=args.adjacent_ops,
                                baseline_score=args.baseline_alignment_score,
                                avoid_delete=False, complete_words=args.complete_words,
                                token_no_space=lambda x: not x.startswith('Ġ'),
                                contiguous_edits=args.contiguous_edits)
        bleu_weights = [1/args.bleu_ngrams if i < args.bleu_ngrams else 0 for i in range(4)]
        env = EditingEnvironment(oracle, n_oracle_edits=args.n_oracle_edits,
                                      oracle_stop_p=args.oracle_stop_p,
                                      bleu_weights = bleu_weights)

        return tokenizer, align_model, oracle, env

class EvaluateBert(Evaluate):

    def setup_env(self, args):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        align_model = BertModel.from_pretrained('bert-base-uncased')
        with open(self.args.idf_path, 'rb') as f:
            idf_dict = pickle.load(f)
        oracle = WordEditOracle(align_model, tokenizer, idf_dict=idf_dict,
                                sort_ops='sort', adjacent_ops=args.adjacent_ops,
                                baseline_score=args.baseline_alignment_score,
                                avoid_delete=False, complete_words=args.complete_words,
                                token_no_space=lambda x: x.startswith('#'),
                                contiguous_edits=args.contiguous_edits)
        bleu_weights = [1/args.bleu_ngrams if i < args.bleu_ngrams else 0 for i in range(4)]
        env = EditingEnvironment(oracle, n_oracle_edits=args.n_oracle_edits,
                                      oracle_stop_p=args.oracle_stop_p,
                                      bleu_weights = bleu_weights)

        return tokenizer, align_model, oracle, env

class EvaluateBartEditor(EvaluateEditor, EvaluateBart):

    def setup_model(self, args):
        model = BertEditor(model_type='bart',
                           tokenizer=self.tokenizer,
                           model_file='facebook/bart-base')
        model.load_state_dict(torch.load(self.args.model_path))
        return model

class EvaluateBertEditor(EvaluateEditor, EvaluateBert):

    def setup_model(self, args):
        model = BertEditor(model_type='bert',
                           tokenizer=self.tokenizer)
        model.load_state_dict(torch.load(self.args.model_path))
        return model


class EvaluateBartS2S(EvaluateS2S, EvaluateBart):

    def setup_model(self, args):
        model = BartS2SEditor(self.align_model, self.tokenizer)
        model.load_state_dict(torch.load(self.args.model_path))
        return model

class EvaluateBartLarge(EvaluateEditor, EvaluateBart):

    def setup_model(self, args):
        model = BertEditor(model_type='bart-large',
                           tokenizer=self.tokenizer,
                           model_file='facebook/bart-large')
        model.load_state_dict(torch.load(self.args.model_path))
        return model

class EvaluateNoModel(EvaluateBartEditor):

    def gen_model(self, canvases):
        return canvases


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_barteditor = subparsers.add_parser('BartEditor')
    parser_barteditor.set_defaults(func=EvaluateBartEditor)
    EvaluateBartEditor.add_args(parser_barteditor)

    parser_barts2s = subparsers.add_parser('BartS2S')
    parser_barts2s.set_defaults(func=EvaluateBartS2S)
    EvaluateBartS2S.add_args(parser_barts2s)

    parser_berteditor = subparsers.add_parser('BertEditor')
    parser_berteditor.set_defaults(func=EvaluateBertEditor)
    EvaluateBertEditor.add_args(parser_berteditor)

    parser_bartlargeeditor = subparsers.add_parser('BartLargeEditor')
    parser_bartlargeeditor.set_defaults(func=EvaluateBartLarge)
    EvaluateBartLarge.add_args(parser_bartlargeeditor)

    parser_baseline = subparsers.add_parser('Baseline')
    parser_baseline.set_defaults(func=EvaluateNoModel)
    EvaluateNoModel.add_args(parser_baseline)

    args = parser.parse_args()
    eval_instance = args.func()
    eval_instance.setup(args)
    eval_instance.run()
