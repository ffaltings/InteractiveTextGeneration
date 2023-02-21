import torch

def compute_entropy(probs):
    log_probs = torch.log(probs)
    entropies = -torch.matmul(probs, log_probs.transpose(0,1)).diagonal()
    return entropies

def top_p_warp(scores, top_p=0.95, filter_value=-float("Inf")):
    sorted_logits, sorted_indices = torch.sort(scores, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores
