import torch

from nltk.tokenize import sent_tokenize

def prep_data_inst(tokenizer,
                   data_inst,
                   n_article_sentences = 3,
                   prompt_length=10,
                   use_highlights=True):
    highlight_prefix = 'Summary: '
    article_prefix = 'Article: '
    highlights = highlight_prefix + ' '.join(data_inst['highlights'].split('\n')) + '\n'
    highlight_ids = tokenizer(highlights, return_tensors='pt')['input_ids'].squeeze()
    article_text = ' '.join(sent_tokenize(data_inst['article'])[:n_article_sentences]) + '\n'
    article_ids = tokenizer(article_text, return_tensors='pt')['input_ids'].squeeze()
    article_prefix_length = 0
    if use_highlights: 
        article_text = article_prefix + article_text
        article_length_no_prefix = article_ids.size(0)
        article_ids = tokenizer(article_text, return_tensors='pt')['input_ids'].squeeze()
        article_prefix_length = article_ids.size(0) - article_length_no_prefix
    if use_highlights:
        input_ids = torch.cat((highlight_ids, article_ids[:prompt_length]))
        highlight_length = highlight_ids.size(0)
    else:
        input_ids = article_ids[:prompt_length]
        highlight_length = 0
    target_ids = article_ids
    return input_ids, target_ids, highlight_length, article_prefix_length
