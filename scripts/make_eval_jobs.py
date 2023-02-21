import argparse
import pickle
import itertools
import json
import os
import re

from distutils.util import strtobool

"""
Expects the following directory structure:
    model_dir /
        exp_name /
            model_name /
                WEIGHTS.bin
            model_name
       exp_name /
    data_dir /
        cnn /
            filtered_bart_64.pickle # cnn dataset
        misc /
            cnn_bart_idfs.pickle # cnn idf scores

Will write the argument files under job_dir, and the scripts will write results to out_dir/job_name. Pass the argument files to the run_eval.py script
to run the experiments.
"""

command_template = '{} \
--model_path {} \
--out_path {} \
--data_path {} \
--max_data {} \
--idf_path {} \
--n_episodes {} \
--n_oracle_edits {} \
--adjacent_ops {} \
--complete_words {} \
--contiguous_edits {} \
--bleu_ngrams {}'

def job_args_from_job_name(job_name):
    if 'bert' in job_name:
        if 'cnn' in job_name:
            data_path_ = os.path.join(data_dir, 'cnn', 'filtered_bert_64')
            idf_path_ = os.path.join(data_dir, 'misc', 'cnn_bert_idfs.pickle')
        elif 'yelp' in job_name:
            data_path_ = os.path.join(data_dir, 'yelp_pe', 'bert_gen_100')
            idf_path_ = os.path.join(data_dir, 'misc', 'yelp_bert_idfs.pickle')
    elif 'bart' in job_name:
        if 'cnn' in job_name:
            data_path_ = os.path.join(data_dir, 'cnn', 'filtered_bart_64')
            idf_path_ = os.path.join(data_dir, 'misc', 'cnn_bart_idfs.pickle')
        elif 'yelp' in job_name:
            data_path_ = os.path.join(data_dir, 'yelp_pe', 'bart_gen_100')
            idf_path_ = os.path.join(data_dir, 'misc', 'yelp_bart_idfs.pickle')

    if 'bert' in job_name:
        func_ = 'BertEditor'
    elif 'bart-s2s' in job_name:
        func_ = 'BartS2S'
    elif 'bart_editor_large' in job_name:
        func_ = 'BartLargeEditor'
    elif 'bart' in job_name:
        func_ = 'BartEditor'

    return data_path_, idf_path_, func_

def make_interactive_eval_job(name, func, model_path, idf_path):
    out_dir_ = os.path.join(int_out_dir, name)
    if not os.path.exists(out_dir_):
        os.makedirs(out_dir_)
    n_episodes_ = [1,2,3]
    n_oracle_edits_ = [6,3,2]
    arg_tuples = []

    for n_ep, n_oe in zip(n_episodes_, n_oracle_edits_):
        out_path_ = os.path.join(out_dir_, f"{n_ep}x{n_oe}")
        arg_tuple = (
            func, model_path, out_path_,
            data_path, max_data, idf_path,
            n_ep, n_oe,
            adjacent_ops, complete_words, contiguous_edits,
            bleu_ngrams)
        arg_tuples.append(arg_tuple)

    job_path = os.path.join(int_job_dir, name)
    with open(job_path, 'wt') as f:
        for a in arg_tuples:
            f.write(command_template.format(*a) + '\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_dir', type=str,
                    default='') # where to write job args
    parser.add_argument('--model_dir', type=str,
                        default='') # where to find model checkpoints
    parser.add_argument('--out_dir', type=str,
                        default='') # where evaluation jobs should write results
    parser.add_argument('--data_dir', type=str,
                        default='') # data dir
    args = parser.parse_args()

    # MAIN ============================

    ## DEFAULT SETTINGS

    job_dir = os.path.join(args.job_dir, 'main')
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    model_dir = args.model_dir
    exp_name = 'dagger'
    model_name = 'cnn-bart-editor'

    func = 'BartEditor'
    model_path = os.path.join(model_dir, exp_name, model_name, 'WEIGHTS.bin')
    out_dir = os.path.join(args.out_dir, 'main')
    out_path = os.path.join(out_dir, 'main.pickle')
    data_dir = args.data_dir
    data_path = os.path.join(data_dir, 'cnn', 'filtered_bart_64')
    max_data = 2000
    idf_path = os.path.join(data_dir, 'misc', 'cnn_bart_idfs.pickle')
    n_episodes = 4
    n_oracle_edits = 3
    adjacent_ops = True
    complete_words = True
    contiguous_edits = True
    bleu_ngrams = 1

    ## Different Oracles

    out_dir_ = os.path.join(out_dir, 'oracles')
    if not os.path.exists(out_dir_):
        os.makedirs(out_dir_)
    oracle_names = ['unrestricted', 'contiguous', 'adjacent']
    oracle_settings = [(False, False), (False, True), (True, False)]
    arg_tuples = []
    for (adjacent_ops_, contiguous_edits_), job_name in zip(oracle_settings, oracle_names):
        out_path_ = os.path.join(out_dir_, job_name)
        arg_tuple = (
            func, model_path, out_path_,
            data_path, max_data, idf_path,
            n_episodes, n_oracle_edits, 
            adjacent_ops_, complete_words, contiguous_edits_,
            bleu_ngrams)
        arg_tuples.append(arg_tuple)
        
    job_path = os.path.join(job_dir, 'oracles')
    with open(job_path, 'wt') as f:
        for a in arg_tuples:
            f.write(command_template.format(*a) + '\n')


    ## DAGGER

    model_names = os.listdir(os.path.join(model_dir, exp_name))
    block_list = []
    model_names = [n for n in model_names if n not in block_list]

    out_dir_ = os.path.join(out_dir, exp_name)
    if not os.path.exists(out_dir_):
        os.makedirs(out_dir_)
    arg_tuples = []
    for job_name in model_names:
        model_path_ = os.path.join(model_dir, exp_name, job_name, 'WEIGHTS.bin')
        out_path_ = os.path.join(out_dir_, job_name)
        
        data_path_, idf_path_, func_ = job_args_from_job_name(job_name) #<= placeholder because name format incorrect for debug
            
        arg_tuple = (
            func_, model_path_, out_path_,
            data_path_, max_data, idf_path_,
            n_episodes, n_oracle_edits,
            adjacent_ops, complete_words, contiguous_edits,
            bleu_ngrams)
        arg_tuples.append(arg_tuple)
    #baseline
    arg_tuples.append(('Baseline', model_path, os.path.join(out_dir_, 'baseline'),
                       data_path, max_data, idf_path, n_episodes, n_oracle_edits,
                       adjacent_ops, complete_words, contiguous_edits, bleu_ngrams))

    job_strings = [command_template.format(*a) for a in arg_tuples]

    job_path = os.path.join(job_dir, exp_name)
    with open(job_path, 'wt') as f:
        for line in job_strings:
            f.write(line + '\n')

    # Interactive vs. One Shot ===========================

    int_job_dir = os.path.join(args.job_dir, 'interactive')
    if not os.path.exists(int_job_dir):
        os.makedirs(int_job_dir)
    int_out_dir = os.path.join(out_dir, 'interactive')
    int_model_dir = os.path.join(model_dir, 'dagger') # evaluation done with dagger models

    names = [
         'cnn-bart-editor', 'cnn-bart_editor_large',
             'cnn-bart-s2s'
    ]
    funcs = [
         'BartEditor', 'BartLargeEditor',
        'BartS2S'
    ]
    idf_names = [
         'cnn_bart_idfs', 'cnn_bart_idfs',
        'cnn_bart_idfs'
    ]

    int_job_args = []
    for n,f,idf_n in zip(names, funcs, idf_names):
        mp = os.path.join(int_model_dir, n, 'WEIGHTS.bin')
        ip = os.path.join(data_dir, 'misc', idf_n + '.pickle')
        int_job_args.append((n,f,mp,ip))

    for a in int_job_args:
        make_interactive_eval_job(*a)
