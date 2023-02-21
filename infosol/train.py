import os
import shutil
import wandb
import pickle
import json
import random
import tqdm
import torch
import itertools
import numpy as np
import argparse
import datetime

from distutils.util import strtobool
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import DatasetDict, Dataset, load_from_disk, concatenate_datasets, load_dataset
from transformers import BertModel, AutoTokenizer, BartModel
from infosol.models.word_edit_model import BertEditor, sample_trajectory, BartS2SEditor
from infosol.env import WordEditOracle, EditingEnvironment
from infosol.alignment import *

DATA_DIR = #/path/to/data_dir/
SAVE_DIR = #/path/to/default/save_dir/
IDF_PATH = os.path.join(DATA_DIR, 'misc', 'cnn_bart_idfs.pickle')
TF_PATH = os.path.join(DATA_DIR, 'misc', 'cnn_bart_tfs.pickle')
DATA_PATH = os.path.join(DATA_DIR, 'cnn', 'filtered_bart_64')

def custom_cat_datasets(data1, data2):
    """
    Utility to concatenate two huggingface datasets
    """
    l1, l2 = len(data1), len(data2)
    data1 = data1.to_dict()
    data2 = data2.to_dict()
    keys = set(data1.keys()).union(set(data2.keys()))
    cat_data= {}
    for k in list(keys):
        d1 = data1.get(k)
        if d1 is None:
            d1 = [None] * l1
        d2 = data2.get(k)
        if d2 is None:
            d2 = [None] * l2
        cat_data[k] = d1 + d2
    return Dataset.from_dict(cat_data)

class WBLogger():

    def log(self, metrics):
        wandb.log(metrics)

class Train():

    """
    Base training job. Doesn't use any generations from model
    """

    def __init__(
        self,
        config=None,
        log_dir=None,
        save_dir=SAVE_DIR,
        rng=None,
        project_name=None,
        run_name=None,
        accumulation_steps=8,
        learning_rate=1e-4,
        n_epochs=1,
        report_every=50,
        val_every=1000,
        device=torch.device('cpu'),
        resume_from_epoch=0,
        keep_token_type_ids=True,
        track_gradient_norm=False,
        clip_grad_norm=False,
        max_grad_norm=2000,
        **kwargs):

        self.config = config
        self.project_name = project_name
        self.run_name = run_name
        self.rng = rng
        self.save_dir=save_dir
        self.model_save_path = os.path.join(self.save_dir, 'WEIGHTS.bin')
        self.log_dir=log_dir
        self.accumulation_steps = accumulation_steps
        self.n_epochs = n_epochs
        self.report_every = report_every
        self.val_every = val_every
        self.device = device
        self.resume_from_epoch = resume_from_epoch
        self.keep_token_type_ids = keep_token_type_ids
        self.track_gradient_norm = track_gradient_norm
        self.clip_grad_norm = clip_grad_norm
        self.max_grad_norm = max_grad_norm

#        self.logger = SummaryWriter(log_dir = self.log_dir)
        self.logger = WBLogger()

        self.kwargs = kwargs

        print("Loading environment")
        self.load_env(**kwargs)
        print("Loading data")
        self.load_data(**kwargs)
        print("Loading model")
        self.load_model(**kwargs)
        print("Loading optimizer")
        self.load_optimizer(**kwargs)

    def log(self, logdict, prefix, i, iter_name='iter'):
        for n in logdict:
            metrics = {
                '/'.join((prefix,n)): np.mean(logdict[n]) for n in logdict}
            metrics[iter_name] = i
            self.logger.log(metrics)
#            self.logger.add_scalar('/'.join((prefix, n)),
#                                   np.mean(logdict[n]),
#                                   i)

    def load_env(self,
                 env_type='bart',
                 idf_path=IDF_PATH,
                 sort_ops='sort',
                 adjacent_ops=False,
                 avoid_delete=False,
                 contiguous_edits=False,
                 complete_words=True,
                 baseline_score=0.3,
                 oracle_stop_p=0.25,
                 n_oracle_hints=-1,
                **kwargs):
        print(f'n_oracle_hints={n_oracle_hints}')

        with open(idf_path, 'rb') as f:
            idf_dict = pickle.load(f)

        if env_type == 'bert':
            self.align_model = BertModel.from_pretrained('bert-base-uncased')
            self.align_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            token_no_space = lambda x: x.startswith('#')
        elif env_type == 'bart':
            self.align_model = BartModel.from_pretrained('facebook/bart-base').encoder
            self.align_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
            token_no_space=lambda x: not x.startswith('Ġ')
        else:
            raise ValueError('Unknown env type: {}'.format(env_type))

        self.oracle = WordEditOracle(self.align_model, self.align_tokenizer, idf_dict,
                                sort_ops=sort_ops, adjacent_ops=adjacent_ops,
                                avoid_delete=avoid_delete, baseline_score=baseline_score,
                                contiguous_edits=contiguous_edits, complete_words=complete_words,
                                token_no_space=token_no_space)
        self.env = EditingEnvironment(self.oracle, oracle_stop_p, n_oracle_edits=n_oracle_hints)

    def load_data(self,
                  data_path=DATA_PATH,
                  batch_size=16,
                  max_train_edits=0,
                  max_val_edits=1000,
                 **kwargs):
        self.data = load_from_disk(data_path)
        self.batch_size = batch_size

        max_train_edits = len(self.data['train']) if max_train_edits == 0 else max_train_edits
        max_val_edits = len(self.data['val']) if max_val_edits == 0 else max_val_edits

        self.data['train'] = self.data['train'].shuffle(generator=self.rng).select(list(range(max_train_edits)))
        self.data['val'] = self.data['val'].shuffle(generator=self.rng).select(list(range(max_val_edits)))

    def load_model(self,
                  tf_path=TF_PATH,
                  model_name='bart',
                  noise_frac=0.0,
                  resume_from_ckpt=None,
                  **kwargs):
        with open(tf_path, 'rb') as f:
            tf_dict = pickle.load(f)
        tf_map, tf_weights = {}, []
        for i,k in enumerate(tf_dict):
            tf_map[i] = k
            tf_weights.append(tf_dict[k])
        vocab_sampler = VocabSampler(tf_weights, tf_map)

        self.model_name = model_name
        if model_name == 'bert':
            self.model = BertEditor(
                               tokenizer=self.align_tokenizer,
                               vocab_sampler=vocab_sampler,
                               training_noise=noise_frac)
        elif model_name == 'bart':
            self.model = BertEditor(
                               tokenizer=self.align_tokenizer,
                               vocab_sampler=vocab_sampler,
                               training_noise=noise_frac,
                               model_type='bart',
                               model_file='facebook/bart-base')
        elif model_name == 'bart-large':
            self.model = BertEditor(
                               tokenizer=self.align_tokenizer,
                               vocab_sampler=vocab_sampler,
                               training_noise=noise_frac,
                               model_type='bart',
                               model_file='facebook/bart-large')
        elif model_name == 'barts2s':
            self.model = BartS2SEditor(self.align_model, self.align_tokenizer)
        elif model_name == 'barts2s-large':
            self.model = BartS2SEditor(self.align_model, self.align_tokenizer, model_file='facebook/bart-large')
        else:
            raise NotImplementedError(f'Unknown model name: {model_name}')

        if not resume_from_ckpt is None:
            print(f'loading model weights from ckpt {resume_from_ckpt}')
            self.model.load_state_dict(torch.load(resume_from_ckpt))

        self.model = self.model.to(self.device)
        self.model = self.model.train()

    def load_optimizer(self, learning_rate=1e-4, **kwargs):
        if self.model_name in ('bert', 'bart', 'bart-large'):
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        elif self.model_name == 'barts2s':
            self.optimizer = torch.optim.Adam(params=self.model.bart_model.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError(f'Unknown model name: {model_name}')

    def pre_train(self):
        self.train_loader = DataLoader(self.data['train'], collate_fn = self.prep_batch, batch_size=self.batch_size,
                                       shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.data['val'], collate_fn = self.prep_batch, batch_size=self.batch_size,
                                     shuffle=False, drop_last=False)

    def train(self):

        def val_(min_val_loss):
            val_loss, val_metrics = self.validate(cur_iter)
            self.log(val_metrics, 'val', cur_iter+1)
            if val_loss <= min_val_loss:
                 torch.save(self.model.state_dict(), self.model_save_path)
                 min_val_loss = val_loss
            return min_val_loss

        self.pre_train()
        min_val_loss = 1e10
        print("Training")
        cur_iter = 0
        batch_num = 0
        with wandb.init(config=self.config, project=self.project_name, name=self.run_name, dir=self.log_dir) as wandb_run:
            wandb.watch(self.model, log='all')
            for e in range(self.resume_from_epoch, self.n_epochs):
                print("Starting epoch: {}".format(e))
                self.optimizer.zero_grad()
                metrics = {}
                for i,batch in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                    batch_num += 1
                    metrics = self.train_step(batch, metrics)
                    if (i+1) % self.accumulation_steps == 0:
                        cur_iter += 1
                        if self.clip_grad_norm:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.max_grad_norm)
                        if self.track_gradient_norm:
                            gradient_norm = 0
                            for p in self.model.parameters():
                                if not p.requires_grad: continue
                                gradient_norm += torch.square(p.grad).sum().item()
                            gradient_norm = np.sqrt(gradient_norm)
                            self.log({'grad_norm': gradient_norm}, 'train', cur_iter)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if (cur_iter+1) % args.report_every == 0:
                            self.log(metrics, 'train', cur_iter+1)
                        if (cur_iter+1) % args.val_every == 0:
                            min_val_loss = val_(min_val_loss)
    #                        val_loss, val_metrics = self.validate(cur_iter)
    #                        self.log(val_metrics, 'val', cur_iter+1)
    #                        if val_loss <= min_val_loss:
    #                            min_val_loss = val_loss
    #                            torch.save(self.model.state_dict(), self.model_save_path)
                        metrics = {}
                self.post_epoch(e)

            _ = val_(min_val_loss)

    def post_epoch(self, e):
        return

    def train_step(self, batch, metrics):
        loss, batch_metrics = self.model.compute_loss(*batch) #TODO: logging
        for n in batch_metrics:
            if n not in metrics:
                metrics[n] = []
            metrics[n].append(batch_metrics[n])
        loss /= self.accumulation_steps
        loss.backward()
        return metrics
    
    def validate(self, cur_iter):
        self.model.eval()
        running_loss = 0
        n_batches = 0
        metrics = {}
        for batch in self.val_loader: # TODO: <= make dataloader
            batch = self.model.move_batch(batch, device)
            with torch.no_grad():
                loss, batch_metrics = self.model.compute_loss(*batch)
            for n in batch_metrics:
                if n not in metrics:
                    metrics[n] = []
                metrics[n].append(batch_metrics[n])
            running_loss += loss.item()
            n_batches += 1
#            for n,l in zip(val_metrics, (loss, e_loss, l_loss, o_loss, v_loss)):
#                val_metrics[n].append(l.item())
#            batch = self.model.move_batch(batch, torch.device('cpu'))

        print("============> Val loss, epoch: iteration: {}".format(cur_iter))
        print(running_loss/n_batches)
#        for n in val_metrics:
#            print("{}: {}".format(n, np.mean(val_metrics[n])))
#            tb_writer.add_scalar('/'.join(('Val', n)), np.mean(val_metrics[n]), cur_iter)
        self.model.train()
        return running_loss/n_batches, metrics


    def prep_batch(self, batch):
        alignments = []
        for b in batch:
            if self.keep_token_type_ids:
                token_type_ids = b.get('token_type_ids')
            else:
                token_type_ids = None
            alignment = Alignment(b['alignment'], b['alignment_scores'], token_type_ids)
            alignment = self.env.oracle_edit(alignment=alignment, device=self.device, return_alignment=True) # TODO: move to sampling
            alignments.append(alignment)
        batch = self.model.prep_batch(alignments, device=self.device)
        return batch

    @classmethod
    def add_args(self, parser):
        parser.add_argument('--accumulation_steps', type=int, default=8)
        parser.add_argument('--n_epochs', type=int, default=20)
        parser.add_argument('--report_every', type=int, default=50)
        parser.add_argument('--val_every', type=int, default=1000)
        parser.add_argument('--resume_from_ckpt', type=str)
        parser.add_argument('--resume_from_epoch', type=int, default=0)
        parser.add_argument('--track_gradient_norm', type=lambda x:bool(strtobool(x)), default=False)
        parser.add_argument('--clip_grad_norm', type=lambda x:bool(strtobool(x)), default=False)
        parser.add_argument('--max_grad_norm', type=float, default=1000)

        data_group = parser.add_argument_group('data')
        data_group.add_argument('--batch_size', type=int, default=16)
        data_group.add_argument('--gen_batch_size', type=int, default=16)
        data_group.add_argument('--max_train_edits', type=int, default=500)
        data_group.add_argument('--max_val_edits', type=int, default=1000)
        data_group.add_argument('--data_path', type=str, default=
                                '/data/scratch/faltings/data/infosol/cnn/filtered_bart_64')
        data_group.add_argument('--keep_token_type_ids', type=lambda x:bool(strtobool(x)), default=True)

        env_group = parser.add_argument_group('env')
        env_group.add_argument('--env_type', type=str, default='bart')
        env_group.add_argument('--oracle_stop_p', type=float, default=0.25)
        env_group.add_argument('--n_oracle_hints', type=int, default=-1)
        env_group.add_argument('--idf_path', type=str, default=
                               '/data/scratch/faltings/data/infosol/misc/cnn_bart_idfs.pickle')
        env_group.add_argument('--n_return_actions', type=int, default=1)
        env_group.add_argument('--sort_ops', type=str, default='sort')
        env_group.add_argument('--avoid_delete', type=lambda x:bool(strtobool(x)), default=False)
        env_group.add_argument('--adjacent_ops', type=lambda x:bool(strtobool(x)), default=False)
        env_group.add_argument('--contiguous_edits', type=lambda x:bool(strtobool(x)), default=False)
        env_group.add_argument('--complete_words', type=lambda x:bool(strtobool(x)), default=True)

        model_group = parser.add_argument_group('model')
        model_group.add_argument('--model_name', type=str, default='bart')
        model_group.add_argument('--noise_frac', type=float, default=0.3)
        model_group.add_argument('--max_traj_length', type=int, default=64)
        model_group.add_argument('--tf_path', type=str, default=
                                 '/data/scratch/faltings/data/infosol/misc/cnn_bart_tfs.pickle')

        opt_group = parser.add_argument_group('optimizer')
        opt_group.add_argument('--learning_rate', type=float, default=1e-4)

        #        parser.add_argument('--n_forward_iterations', type=int, default=1) TODO: ForwarTrain

class GenerationInstance():

    """
    Used for training jobs that need generation
    """

    def __init__(self, datum, env):
        self.done = False
        self.target = datum['target_tokens']
        self.target_text = datum['target_text']
        canvas = Canvas(datum['source_tokens'], datum['token_type_ids'])
        assert (not datum['alignment'] is None) and (not datum['alignment_scores'] is None)
        alignment = Alignment(datum['alignment'], datum['alignment_scores'])
        self.history = [(canvas.copy(), alignment.copy())]
        self.oracle_canvas = env.reset(alignment=alignment)

    def make_data(self, tokenizer):
        instances = []
        for canvas, alignment in self.history:
            inst = {
                'source_text': canvas.render(tokenizer),
                'source_tokens': canvas.tokens,
                'token_type_ids': canvas.type_ids,
                'target_text': self.target_text,
                'target_tokens': self.target,
                'alignment': alignment.alignment,
                'alignment_scores': alignment.scores
            }
            instances.append(inst)
        return instances

class DaggerTrain(Train):

    """
    Dagger training job. Split into epochs that run through small batches of data generated
    from the model or sampled from the dataset.
    """

    def __init__(
        self,
        top_k=10,
        top_p=1.0,
        stop_threshold=0.9,
        max_iter=16,
        do_sample=True,
        max_length=64,
        parallel_decode=False,
        n_processes=10,
        n_warmup_epochs=0,
        sampling_annealing_rate=1.0,
        dagger_sampling_rate=1.0,
        max_trajectory_length=2,
        sample_batch_size=4096,
        val_sample_batch_size=1024,
        sample_val_every_n_epoch=25,
        sample_train_every_n_epoch=10,
        **kwargs):

        super().__init__(**kwargs)

        self.top_k = top_k
        self.top_p = top_p
        self.stop_threshold = stop_threshold
        self.max_iter = max_iter
        self.do_sample = do_sample
        self.max_length = max_length
        self.parallel_decode = parallel_decode
        self.n_processes = n_processes

        self.n_warmup_epochs = n_warmup_epochs
        self.sampling_annealing_rate = sampling_annealing_rate
        self.sample_expert_p = dagger_sampling_rate
        self.dagger_sampling_rate = dagger_sampling_rate # base rate
        self.max_trajectory_length = max_trajectory_length
        self.sample_batch_size = sample_batch_size
        self.sample_val_every_n_epoch = sample_val_every_n_epoch
        self.sample_train_every_n_epoch = sample_train_every_n_epoch
        self.val_sample_batch_size = val_sample_batch_size

        self.model_save_path = os.path.join(self.save_dir, 'WEIGHTS.bin')

    def gen_episode(self, instances):
        """
        Generate one episode of a trajectory
        """
        canvases = [inst.oracle_canvas for inst in instances]
        generations = self.gen_model(canvases)
        clean_generations = [g.clean() for g in generations]
        align_model, align_tokenizer = self.env.oracle.align_model, self.env.oracle.align_tokenizer
        targets = [inst.target for inst in instances]
        alignments = list(batch_align_canvases(clean_generations, targets, align_model, align_tokenizer, device=self.device))
        for i,inst in enumerate(instances):
            inst.history.append((clean_generations[i].copy(), alignments[i].copy()))
            inst.oracle_canvas = self.env.oracle_edit(alignment=alignments[i])
        return instances

    def sample_batch(self, data):

        """
        Sample a new batch of data, i.e. generated trajectories from the current model
        """

        instances = [GenerationInstance(d, self.env) for d in data]
#        for inst in instances:
#            if np.random.random() <= self.sample_expert_p:
#                inst.done=True

        self.env.oracle = self.env.oracle.to(self.device)
        self.model = self.model.eval()
        finished_instances = []
        n_sampled_states = 0
        for i in range(self.max_trajectory_length):
            for inst in instances:
                if np.random.random() <= self.sample_expert_p or len(inst.history) >= self.max_trajectory_length:
                    finished_instances.append(inst)
                    inst.done = True
                    n_sampled_states += len(inst.history)
            instances = [inst for inst in instances if not inst.done]
            # TODO: instead of fixing a maximum trajectory length, fix a max number of instances, then get a distribution over traj lengths
#            if n_sampled_states > self.sample_batch_size or len(instances) == 0:
            if len(instances) == 0:
                break

            instances = self.gen_episode(instances)
        finished_instances.extend(instances)
        self.env.oracle = self.env.oracle.cpu()
        self.model = self.model.train()

        sampled_states = []
        for inst in finished_instances:
            sampled_states.extend(inst.make_data(self.align_tokenizer))
        return sampled_states

    def pre_train(self):
#        self.sampling_rate_updates = 0
        self.post_epoch(-1)

    def post_epoch(self, e):
        # Sample trajectories here!
        def get_batch(data, size):
            idxs = np.random.choice(np.arange(len(data)), size, replace=False)
            for i in idxs:
                yield data[int(i)]

        if e < self.n_warmup_epochs or (e+1) % self.sample_train_every_n_epoch == 0:
            print("Sampling train batch")
            if e >= self.n_warmup_epochs:
#                self.sampling_rate_updates += 1
                self.sample_expert_p *= self.sampling_annealing_rate
#                np.exp(self.sampling_rate_updates * np.log(self.sampling_annealing_rate)) * self.dagger_sampling_rate
                print(self.sample_expert_p)
            train_batch = self.sample_batch(get_batch(self.data['train'], self.sample_batch_size))
            self.train_loader = DataLoader(train_batch, collate_fn = self.prep_batch, batch_size=self.batch_size,
                                           shuffle=True, drop_last=True)

        if (e+1) % self.sample_val_every_n_epoch == 0:
            print("Sampling val batch")
            val_batch = self.sample_batch(get_batch(self.data['val'], self.val_sample_batch_size))
            self.val_loader = DataLoader(val_batch, collate_fn = self.prep_batch, batch_size=self.batch_size,
                                         shuffle=False, drop_last=False)


#    def sample_trajectory(self, datum):
#        alignments = []
#        alignment = Alignment(alignment=datum['alignment'], scores=datum['alignment_scores'])
#        alignment = self.env.reset(alignment=alignment, return_alignment=True)
#        i = 0
#        while True:
#            alignments.append(alignment)
#            if np.random.random() > self.sample_expert_p\
#               and i < self.max_trajectory_length:
#                canvas = alignment.get_source_canvas()
#                canvas = self.gen_model(canvas)
#                alignment,_ = self.env.step(canvas=canvas, return_alignment=True, device=self.device)
#                i += 1
#            else:
#                break
#        return alignments
#
#    def prep_batch(self, batch):
#        alignments = []
#        for b in batch:
#            alignments.extend(self.sample_trajectory(b))
#            if len(alignments) >= self.batch_size: break
#        batch = self.model.prep_batch(alignments, device=self.device)
#        return batch

    def gen_model(self, canvases):
        if self.model_name in ('bart', 'bert', 'bart-large'):
            canvases = list(tqdm.tqdm(self.model.batch_depth_decode(
                canvases,
                top_k=self.top_k,
                max_batch_tokens=2048,
                device=self.device,
#                parallel=self.parallel_decode,
                return_idx=True,
                queue_size=2000,
                max_iter=self.max_iter,
#                n_processes=self.n_processes
            ), total=len(canvases)))
            canvases = [(i,c) for c,i in canvases]
            canvases = [c for i,c in sorted(canvases)]
            return canvases
        elif self.model_name == 'barts2s': #TODO
            canvases = self.model.batch_generate(canvases, device=self.device,
                do_sample=self.do_sample, top_p=self.top_p, max_length=self.max_length)
#            print(canvases)
            return canvases
    
    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)

        model_gen_group = parser.add_argument_group('model generation')
        model_gen_group.add_argument('--top_k', type=int, default=10)
        model_gen_group.add_argument('--top_p', type=float, default=0.95)
        model_gen_group.add_argument('--stop_threshold', type=float, default=0.9)
        model_gen_group.add_argument('--parallel_decode', type=lambda x:bool(strtobool(x)), default=False)
        model_gen_group.add_argument('--n_processes', type=int, default=10)
        model_gen_group.add_argument('--do_sample', type=lambda x:bool(strtobool(x)), default=True)
        model_gen_group.add_argument('--max_length', type=int, default=64)
        model_gen_group.add_argument('--max_iter', type=int, default=32)

        dagger_group = parser.add_argument_group('dagger')
        dagger_group.add_argument('--n_warmup_epochs', type=int, default=0)
        dagger_group.add_argument('--sampling_annealing_rate', type=float, default=1.0)
        dagger_group.add_argument('--dagger_sampling_rate', type=float, default=0.5)
        dagger_group.add_argument('--max_trajectory_length', type=int, default=2)
        dagger_group.add_argument('--sample_batch_size', type=int, default=100)
        dagger_group.add_argument('--sample_val_every_n_epoch', type=int, default=50)
        dagger_group.add_argument('--sample_train_every_n_epoch', type=int, default=15)
        dagger_group.add_argument('--val_sample_batch_size', type=int, default=100)

class ForwardTrain(Train):

    """
    Forward training job. Similar to dagger but generates less frequently.
    Not used anymore
    """

    def __init__(
        self,
        top_k=10,
        top_p=1.0,
        stop_threshold=1.0,
        max_iter=16,
        do_sample=True,
        max_length=64,
        n_forward_iter=2,
        resume_forward_iter=0,
        sample_alg='depth',
        sample_gen_reverse_steps=64,
        force_regen=False,
        n_processes=None,
        parallel_decode=False,
        **kwargs):

        super().__init__(**kwargs)

        self.top_k = top_k
        self.sample_alg = sample_alg
        self.top_p = top_p
        self.stop_threshold = stop_threshold
        self.max_iter = max_iter
        self.do_sample = do_sample
        self.max_length = max_length
        self.sample_gen_reverse_steps = sample_gen_reverse_steps
        self.n_forward_iter = n_forward_iter
        self.resume_forward_iter = resume_forward_iter
        self.force_regen = force_regen
        self.n_processes = n_processes
        self.parallel_decode = parallel_decode

        self.meta_log_dir = self.log_dir
        self.meta_run_name = self.run_name

    def gen_model(self, input_generator):
        if self.model_name in ('bart', 'bart-large', 'bert'):
            if self.sample_alg == 'sample':
                for canvas, idx in self.model.batch_decode(
                    input_generator,
                    top_k = self.top_k,
                    max_batch_tokens=2048,
                    device=self.device,
#                    parallel=self.parallel_decode,
                    return_idx=True,
                    queue_size=2000,
                    max_iter=self.max_iter,
#                    n_processes=self.n_processes
                ):
                    yield canvas, idx
            elif self.sample_alg == 'depth':
                for canvas, idx in self.model.batch_depth_decode(
                    input_generator,
                    top_k = self.top_k,
                    max_batch_tokens=2048,
                    device=self.device,
#                    parallel=self.parallel_decode,
                    return_idx=True,
                    queue_size=2000,
                    max_iter=self.max_iter,
                    stop_threshold=self.stop_threshold,
#                    n_processes=self.n_processes
                ):
                    yield canvas, idx
        elif self.model_name == 'barts2s':
            for i,canvas in enumerate(input_generator):
                canvas = self.model.generate(
                    canvas, device=self.device, do_sample=self.do_sample, top_p=self.top_p, max_length=self.max_length
                )
                yield canvas, i
        return

    def batch_generate(self, data, save_path, chunk_size=10000, **kwargs):# batch_size=1 for now because it is still not scaling up well (because of differences in canvas lengths, etc.)
        def generator_(data, data_buffer):
            for i,datum in enumerate(data):
                if self.keep_token_type_ids:
                    token_type_ids = datum.get('token_type_ids')
                else:
                    token_type_ids = None
                alignment = Alignment(datum['alignment'], datum['alignment_scores'], token_type_ids)
                alignment = self.env.oracle_edit(alignment=alignment, return_alignment=True)

                # push forward
                non_const_ops = alignment.get_non_const_ops()
                n_forward_steps = max(0, len(non_const_ops) - self.sample_gen_reverse_steps)
                forward_ops = np.random.choice(non_const_ops, n_forward_steps)
                alignment.push_forward(forward_ops)
                data_buffer[i] = alignment
                yield alignment.get_source_canvas()

        # chunk generator
        def chunks(iterator, n):
            for first in iterator: # take one item out (exits loop if `iterator` is empty)
                rest_of_chunk = itertools.islice(iterator, 0, n - 1)
                yield itertools.chain([first], rest_of_chunk)  # concatenate the first item back

        self.model = self.model.eval()
        with open(save_path, 'wt') as f:
            for i,chunk in enumerate(chunks(iter(data), chunk_size)):
                print(f'On chunk {i}')
                timeouts = 0
                data_buffer = {}
                popped_idxs = set()
                input_generator = generator_(chunk, data_buffer)
                output_generator = self.gen_model(input_generator)
                for canvas, idx in tqdm.tqdm(output_generator, total=chunk_size):
                    if canvas is None:
                        timeouts += 1
                        print('timeout')
                        continue
                    try:
                        alignment = data_buffer.pop(idx)
                    except KeyError as e:
                        print(np.max(list(data_buffer.keys())))
                        print(idx in popped_idxs)
                        raise e
                    popped_idxs.add(idx)
                    canvas = canvas.clean()
                    json_str = json.dumps({
                        'source_tokens': canvas.tokens,
                        'token_type_ids': canvas.type_ids,
    #                    'idx': idx,
                        'source_text': canvas.render(self.align_tokenizer),
                        'target_tokens': alignment.get_target_tokens(),
                        'target_text': alignment.get_target_canvas().render(self.align_tokenizer),
                    })
                    f.write(json_str + '\n')
                print(f'{timeouts} timeouts')

    def align_batch(self, batch):
        tokens_a = batch['source_tokens']
        tokens_b = batch['target_tokens']
        alignments = list(
            batch_align(
                tokens_a, tokens_b, self.align_model, self.align_tokenizer,
                baseline_score=0.3, device=self.device
            )
        )
        alignment_scores = [a.scores for a in alignments]
        alignments = [a.alignment for a in alignments]
        batch['alignment'] = alignments
        batch['alignment_scores'] = alignment_scores
        return batch

    def load_data(self, **kwargs):
        super().load_data(**kwargs)
        self.gen_data = self.data

    def pre_train(self):
        self.train_loader = DataLoader(self.data['train'], collate_fn = self.prep_batch, batch_size=self.batch_size,
                                       shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.data['val'], collate_fn = self.prep_batch, batch_size=self.batch_size,
                                     shuffle=False, drop_last=False)

    def train(self):
        if self.resume_forward_iter > 0:
            forward_iter = self.resume_forward_iter - 1
            iter_save_dir = os.path.join(self.save_dir, f'forward_iter_{forward_iter}')
            for i in range(forward_iter):
                gen_save_path = os.path.join(self.save_dir, f'forward_iter_{i}', 'gen_data')
                self.gen_data = load_from_disk(gen_save_path)
                self.add_data(self.gen_data)

            gen_save_path = os.path.join(iter_save_dir, 'gen_data')
            if not os.path.exists(gen_save_path) or self.force_regen:
                weights_path = os.path.join(iter_save_dir, 'WEIGHTS.bin')
                self.kwargs.update({'resume_from_ckpt': weights_path})
                self.load_model(**self.kwargs)
                self.kwargs.update({'resume_from_ckpt': None})
                self.generate(forward_iter, iter_save_dir)
            else:
                self.gen_data = load_from_disk(gen_save_path)
                self.add_data(self.gen_data)

            self.load_model(**self.kwargs)
            self.load_optimizer(**self.kwargs)

        for forward_iter in range(self.resume_forward_iter, self.n_forward_iter):

            iter_save_dir = os.path.join(self.save_dir, f'forward_iter_{forward_iter}')
            if not os.path.exists(iter_save_dir):
                os.makedirs(iter_save_dir)
            self.model_save_path = os.path.join(iter_save_dir, 'WEIGHTS.bin')

            iter_logdir = os.path.join(self.meta_log_dir, f'forward_iter_{forward_iter}')
            if not os.path.exists(iter_logdir):
                os.makedirs(iter_logdir)
            self.log_dir = iter_logdir

            self.run_name = '-'.join((self.meta_run_name, f'forward_iter_{forward_iter}'))
#            self.logger = SummaryWriter(log_dir = iter_logdir)

            super().train()

            if forward_iter == (self.n_forward_iter - 1):
                break

            # load best checkpoint
            self.kwargs.update({'resume_from_ckpt': self.model_save_path})
            self.load_model(**self.kwargs)
            self.kwargs.update({'resume_from_ckpt': None})

            self.generate(forward_iter, iter_save_dir)

            print('reloading model')

            self.load_model(**self.kwargs)
            self.load_optimizer(**self.kwargs)

    def generate(self, forward_iter, iter_save_dir):

        print('generating')
        train_save_path = os.path.join(iter_save_dir, 'train_data')
        val_save_path = os.path.join(iter_save_dir, 'val_data')
        self.batch_generate(self.gen_data['train'], train_save_path)
        self.batch_generate(self.gen_data['val'], val_save_path)

        def listdict2dictlist(ld):
            keys = ld[0].keys()
            return {
                k: [d[k] for d in ld] for k in keys
            }

        train_generations, val_generations = [], []
        with open(train_save_path, 'rt') as f:
            for line in f:
                train_generations.append(json.loads(line))
        with open(val_save_path, 'rt') as f:
            for line in f:
                val_generations.append(json.loads(line))

        generated_data = DatasetDict({
            'train': Dataset.from_dict(listdict2dictlist(train_generations)),
            'val': Dataset.from_dict(listdict2dictlist(val_generations))
        })

#        generated_data = DatasetDict({
#            'train': load_dataset('json', train_save_path),
#            'val': load_dataset('json', val_save_path)
#        })

        self.align_model = self.align_model.to(self.device)
        print('aligning')
        generated_data = generated_data.map(self.align_batch, batched=True, batch_size=64)
        self.align_model = self.align_model.cpu()
        self.gen_data = generated_data

        os.remove(train_save_path)
        os.remove(val_save_path)

        gen_save_path = os.path.join(iter_save_dir, 'gen_data')
        generated_data.save_to_disk(gen_save_path)

        self.add_data(generated_data)

    def add_data(self, dataset):

        self.data = DatasetDict({
            'train': custom_cat_datasets(dataset['train'], self.data['train']),
            'val': custom_cat_datasets(dataset['val'], self.data['val'])
        })

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)

        parser.add_argument('--sample_alg', choices=['depth', 'sample'], default='depth')
        parser.add_argument('--top_k', type=int, default=10)
        parser.add_argument('--top_p', type=float, default=0.95)
        parser.add_argument('--stop_threshold', type=float, default=0.9)
        parser.add_argument('--max_iter', type=int, default=32)
        parser.add_argument('--do_sample', type=lambda x:bool(strtobool(x)), default=False)
        parser.add_argument('--max_length', type=int, default=64)
        parser.add_argument('--n_forward_iter', type=int, default=2)
        parser.add_argument('--resume_forward_iter', type=int, default=0)
        parser.add_argument('--sample_gen_reverse_steps', type=int, default=64)
        parser.add_argument('--force_regen', type=lambda x:bool(strtobool(x)), default=False)
        parser.add_argument('--parallel_decode', type=lambda x:bool(strtobool(x)), default=False)
        parser.add_argument('--n_processes', type=int)

#class NBTrain(Train):
#
#    def __init__(
#        self,
#        top_k=10):
#
#        self.top_k = top_k



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--use_timestamp', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_dir', type=str, 
                        default='/Mounts/rbg-storage1/users/faltings/cache/infosol/experiment_results/train/')
    parser.add_argument('--run_name', type=str,
                        default='debug')
    parser.add_argument('--project_name', type=str,
                        default='infosol')
    subparsers = parser.add_subparsers()

    base_parser = subparsers.add_parser('base')
    base_parser.set_defaults(func=Train)
    Train.add_args(base_parser)

    dagger_parser = subparsers.add_parser('dagger')
    dagger_parser.set_defaults(func=DaggerTrain)
    DaggerTrain.add_args(dagger_parser)

    forward_parser = subparsers.add_parser('forward')
    forward_parser.set_defaults(func=ForwardTrain)
    ForwardTrain.add_args(forward_parser)

    args = parser.parse_args()

    device=torch.device('cuda:{}'.format(args.cuda_device)) if torch.cuda.is_available() else torch.device('cpu')

    if args.use_timestamp:
        run_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_dir = os.path.join(args.run_dir, args.run_name, run_id)
    else: run_dir = os.path.join(args.run_dir, args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    config_path = os.path.join(run_dir, 'config.conf')
    with open(config_path, 'w') as f:
        for k,v in vars(args).items():
            f.write('--' + k + '\n' + str(v) + '\n')

    log_dir = os.path.join(run_dir, 'logs')
#    if os.path.exists(log_dir):
#        shutil.rmtree(log_dir)
#        for logfile in os.listdir(log_dir):
#            os.remove(os.path.join(log_dir, logfile))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    rng = np.random.default_rng(seed = args.seed)
    train = args.func(
        config = vars(args),
        save_dir = run_dir,
        log_dir = log_dir,
        device = device,
        rng = rng,
        **vars(args))
    train.train()
