import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch.nn as nn
import argparse
import pickle
from torchmetrics.audio import SignalNoiseRatio, ScaleInvariantSignalNoiseRatio
from model import ssnet
from sampler import Samples
from utils import utility_fct

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str)
parser.add_argument('batch_size', type=int)
parser.add_argument('model', type=str)
args = parser.parse_args()

snr = SignalNoiseRatio()

if __name__ == "__main__":

    test = pickle.load(open(args.data_path, 'rb'))
    X_test, y_test = test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssnet_ = ssnet().to(device)
    
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        ssnet_ = nn.DataParallel(ssnet_)
    checkpoint = torch.load(args.model)
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    ssnet_.load_state_dict(state_dict)
    
    test_set = Samples(X_test, y_test)

    test_loader = DataLoader(test_set, collate_fn=utility_fct, batch_size=args.batch_size, num_workers=8)

    results = []
    ssnet_.eval()
    for X, y in test_loader:
        X = X.cuda()
        out = ssnet_(X)[:,:,:,100:-100]
        for out_i, y_i in zip(out,y):
            out_s1, y_s1 = out_i[0,:,:], y_i[0,:,:]
            out_s2, y_s2 = out_i[1,:,:], y_i[1,:,:]
            out_s3, y_s3 = out_i[2,:,:], y_i[2,:,:]
            out_s4, y_s4 = out_i[3,:,:], y_i[3,:,:]
            snr1 = snr(out_s1, y_s1)
            snr2 = snr(out_s2, y_s2)
            snr3 = snr(out_s3, y_s3)
            snr4 = snr(out_s4, y_s4)
            results.append((snr1.cpu().detach().numpy(),snr2.cpu().detach().numpy(),snr3.cpu().detach().numpy(),snr4.cpu().detach().numpy()))
    print(f'SNR for source 1 (drums): {[np.mean(x[0]) for x in results]}')
    print(f'SNR for source 2 (bass): {[np.mean(x[1]) for x in results]}')
    print(f'SNR for source 3 (rest of accompaniment): {[np.mean(x[2]) for x in results]}')
    print(f'SNR for source 4 (vocals): {[np.mean(x[3]) for x in results]}')