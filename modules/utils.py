import torch

def pad_seqs(seqs, num_chan, num_pad=100):
    pad_seqs = []
    for seq in seqs:
        pad = torch.zeros(num_chan, num_pad)
        pad_seq = torch.cat([pad, seq, pad], dim=1)
        pad_seqs.append(pad_seq)
    return torch.stack(pad_seqs, dim=0)

def utility_fct(Xy):
    seq1, seq2 = zip(*Xy)
    X, y = pad_seqs(seq1, 2), pad_seqs(seq2, 2)
    return (X, y)

def transform(x):
    return torch.tensor(x).T.float()