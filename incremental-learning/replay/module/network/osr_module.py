import torch
import torch.nn as nn

def entropy(preds, use_softmax=True):
    if use_softmax:
        preds = torch.nn.Softmax(dim=-1)(preds)

    logp = torch.log(preds + 1e-5)
    entropy = torch.sum(-preds * logp, dim=-1)
    return entropy

def msp(preds):
    # preds B x T x C
    preds = torch.nn.Softmax(dim=-1)(preds)
    score, _ = torch.max(preds, dim=-1) # B x T
    return score

def mls(preds):
    # preds B x T x C
    score, _ = torch.max(preds, dim=-1) # B x T
    return score