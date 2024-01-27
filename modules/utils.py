import torch

def pad_seqs(seqs, num_chan, min_pad=100):
    pad_seqs = []
    max_len = max([x.shape[1] for x in seqs])+min_pad
    for seq in seqs:
        diff_len = max_len - seq.shape[1]
        padL, padR = torch.zeros(num_chan, diff_len//2), torch.zeros(num_chan, diff_len//2+diff_len%2)
        pad_seq = torch.cat([padL, seq, padR], dim=1)
        pad_seqs.append(pad_seq)
    return torch.stack(pad_seqs, dim=0)

def utility_fct(Xy):
    seq1, seq2 = zip(*Xy)
    X, y = pad_seqs(seq1, 4), pad_seqs(seq2, 1)
    return (X, y)

def get_loss(X, y, out, loss_fct):
    loss = loss_fct(out, y).cuda()
    zero_mask = torch.all(X == 0, dim=1)
    zero_mask = zero_mask.unsqueeze(1)
    zero_mask = zero_mask.expand(-1, 1, -1)
    loss[zero_mask] = 0.
    lens = torch.sum(X, dim=(1,-1))
    loss_sums = torch.sum(loss, dim=(1,-1))
    return (loss_sums/lens).mean()