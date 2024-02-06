import torch
from scipy.fft import fft
import numpy as np

#add padding on both sides of samples
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
    x = fft(x)
    return torch.tensor(x).float()

def transform(x):
    x_fft = fft(x)
    magnitude = np.abs(x_fft)
    log_magnitude = np.log(magnitude + 1e-6)
    return torch.tensor(log_magnitude).float()

def transform(x):
    x_fft = fft(x)
    
    x_fft_tensor = torch.tensor(x_fft, dtype=torch.complex64)
    
    magnitude = torch.abs(x_fft_tensor)
    
    log_magnitude = torch.log1p(magnitude)
    
    return log_magnitude.float()

