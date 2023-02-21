import sys
import os
import json
import torch
import itertools
import tqdm
import pickle
import random
import argparse
import numpy as np
import matplotlib.pyplot as pltq

from sklearn.feature_extraction.text import CountVectorizer
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, BertModel, BartModel, AutoModel
from infosol.alignment import batch_align, get_non_const_ops, sample_actions
from nltk.tokenize import sent_tokenize

from transformers import AutoModel
model = AutoModel.from_pretrained('facebook/bart-large')
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

def tokenize_text(data_inst):
    data_inst['target_tokens'] = tokenizer.tokenize(data_inst['target_text'])
    data_inst['source_tokens'] = tokenizer.tokenize(data_inst['source_text'])
    return data_inst

def make_len_filter(max_len=32, min_len=0):
    def filter_(data_inst):
        return len(data_inst['target_tokens']) <= max_len and len(data_inst['target_tokens']) >= min_len
    return filter_

def align_batch(data_batch, model, tokenizer, baseline_score=0.3, device=torch.device('cpu')):
    tokens_a = data_batch['source_tokens']
    tokens_b = data_batch['target_tokens']
    alignments = list(batch_align(tokens_a, tokens_b,
        model, tokenizer, baseline_score=baseline_score, device=device))
    alignment_scores = [a.scores for a in alignments]
    alignments = [a.alignment for a in alignments]
    data_batch['alignment'] = alignments
    data_batch['alignment_scores'] = alignment_scores
    return data_batch

def is_edit(data_inst):
    return len(get_non_const_ops(data_inst['alignment'])) > 0

def compute_tf_idf_scores(data, tokenizer):
    sentences = [datum['target_text'] for datum in data]
    cvectorizer = CountVectorizer(analyzer=tokenizer.tokenize)
    counts = cvectorizer.fit_transform(sentences)
    
    document_counts = counts > 0

    doc_counts_sum = document_counts.sum(axis=0)
    counts_sum = counts.sum(axis=0)

    idf_scores = {k: -np.log(doc_counts_sum[0,cvectorizer.vocabulary_[k]]) for k in tqdm.tqdm(cvectorizer.vocabulary_)}
    idf_scores = {k: np.log(document_counts.shape[0]) + idf_scores[k] for k in idf_scores}

    total = counts_sum.sum()
    tf_dict = {k: counts_sum[0, cvectorizer.vocabulary_[k]]/total for k in cvectorizer.vocabulary_}

    return idf_scores, tf_dict

def add_type_ids(datum):
    datum['token_type_ids'] = [0] * len(datum['source_tokens'])
    return datum

def remove_type_ids(datum):
    datum.pop('token_type_ids')
    return datum

def tokenize_text(data_inst, tokenizer):
    data_inst['target_tokens'] = tokenizer.tokenize(data_inst['target_text'])
    data_inst['source_tokens'] = tokenizer.tokenize(data_inst['source_text'])
    return data_inst

def random_subset(data, size, seed=42):
    return data.shuffle(
        generator=np.random.default_rng(seed)
    ).select(
        range(size)
    )

def random_subset_dict(dataset_dict, split_sizes, seed=42):
    return DatasetDict({
        n: random_subset(dataset_dict[n], s, seed=seed) for n,s in zip(dataset_dict, split_sizes)
    })

def process_cnn_instances(data_inst, tokenizer):
    return_instances = []
    for sent in data_inst['highlights'][0].split('\n'):
        return_inst = {}
        target_text = sent
        target_tokens = tokenizer.tokenize(target_text)
        alignment = [('', t) for t in target_tokens]
        alignment_scores = [0.3] * len(alignment)
        source_tokens = []
        source_text = ''
        token_type_ids = []
        return_inst = {
            'id': data_inst['id'][0],
            'target_text': target_text,
            'target_tokens': target_tokens,
            'source_text': source_text,
            'source_tokens': source_tokens,
            'alignment': alignment,
            'alignment_scores': alignment_scores,
            'token_type_ids': token_type_ids
        }
        return_instances.append(return_inst)
    return_instances = {k: [inst[k] for inst in return_instances] for k in return_instances[0]}
    return return_instances

def make_cnn_data(tokenizer):
    data = load_dataset('cnn_dailymail', '3.0.0')
    data = data.map(lambda x: process_cnn_instances(x, tokenizer), batched=True, batch_size=1, remove_columns=['article', 'highlights', 'id'])
    data = data.filter(make_len_filter(max_len=64, min_len=10))
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='') # where to save data
    args = parser.parse_args()

    bart_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    bart_model = BartModel.from_pretrained('facebook/bart-base').encoder
    device = torch.device('cuda')
    bart_model = bart_model.to(device)

    cnn_data = load_dataset('ccdv/cnn_dailymail', '3.0.0')

    bart_cnn_data = make_cnn_data(bart_tokenizer)

    save_path = os.path.join(args.data_dir, 'cnn_bart')
    bart_cnn_data.save_to_disk(save_path)

    cnn_bart_idfs, cnn_bart_tfs = compute_tf_idf_scores(bart_cnn_data['train'].select(range(500000)), bart_tokenizer)

    misc_dir = os.path.join(args.data_dir, 'misc')
    if not os.path.exists(misc_dir):
        os.makedirs(misc_dir)

    bart_idf_save_path = os.path.join(misc_dir, 'cnn_bart_idfs.pickle')
    with open(bart_idf_save_path, 'wb') as f:
        pickle.dump(cnn_bart_idfs, f)

    bart_tf_save_path = os.path.join(misc_dir, 'cnn_bart_tfs.pickle')
    with open(bart_tf_save_path, 'wb') as f:
        pickle.dump(cnn_bart_tfs, f)
