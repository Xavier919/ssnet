import torch

def pad_seqs(seqs, num_chan, num_pad=100):
    pad = torch.zeros(num_chan, num_pad)
    pad_seqs = [torch.cat([pad, x, pad], dim=1) for x in seqs]
    return torch.stack(pad_seqs, dim=0)

def utility_fct(Xy):
    X, y = zip(*Xy)
    X = pad_seqs(X, 2, num_pad=100)
    y = torch.stack(y, dim=0)
    return (X, y)

def transform(x):
    return torch.tensor(x).float()