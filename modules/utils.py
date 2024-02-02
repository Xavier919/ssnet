import torch
from scipy.fft import fft
import numpy as np

#add padding on both sides of samples
def pad_seqs(seqs, num_chan, num_pad=100):
    pad = torch.zeros(num_chan, num_pad)
    pad_seqs = [torch.cat([pad, x, pad], dim=1) for x in seqs]
    return torch.stack(pad_seqs, dim=0)

def apply_fourier(x):
    x = fft(x)
    x = np.abs(x)
    x = np.log(x + 1e-9)
    return x

def utility_fct(Xy):
    X, y = zip(*Xy)
    X = pad_seqs(X, 2, num_pad=100)
    y = torch.stack(y, dim=0)
    return (apply_fourier(X), apply_fourier(y))

def transform(x):
    x = fft(x)
    x = np.abs(x)
    x = np.log(x + 1e-9)
    return torch.tensor(x).float()