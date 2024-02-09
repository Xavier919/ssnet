import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch.nn as nn
import argparse
import pickle
from torchmetrics.audio import SignalDistortionRatio, SignalNoiseRatio
from model import ssnet
from sampler import Samples
from utils import utility_fct

parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str)
parser.add_argument('batch_size', type=int)
parser.add_argument('kernel', type=int)
parser.add_argument('model', type=str)
args = parser.parse_args()


if __name__ == "__main__":

    test = pickle.load(open(args.data_path, 'rb'))
    X_test, y_test = test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ssnet(k=args.kernel).to(device)
    checkpoint = torch.load(args.model, map_location=device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint)
    else:
        state_dict = {key.replace("module.", ""): value for key, value in checkpoint.items()}
        model.load_state_dict(state_dict)
    
    test_set = Samples(X_test, y_test)

    test_loader = DataLoader(test_set, collate_fn=utility_fct, batch_size=args.batch_size, num_workers=8)

    sdr = SignalDistortionRatio().to(device)

    results = []
    model.eval()
    for X, y in test_loader:
        X = X.cuda()
        out = model(X)[:,:,:,100:-100].cpu().detach()
        for out_i, y_i in zip(out,y):
            out_s1, y_s1 = out_i[0,:,:], y_i[0,:,:].cpu().detach()
            out_s2, y_s2 = out_i[1,:,:], y_i[1,:,:].cpu().detach()
            out_s3, y_s3 = out_i[2,:,:], y_i[2,:,:].cpu().detach()
            out_s4, y_s4 = out_i[3,:,:], y_i[3,:,:].cpu().detach()
            sdr1 = sdr(out_s1, y_s1)
            print(sdr1)
            sdr2 = sdr(out_s2, y_s2)
            print(sdr2)
            sdr3 = sdr(out_s3, y_s3)
            print(sdr3)
            sdr4 = sdr(out_s4, y_s4)
            print(sdr4)
            results.append((sdr1.item(),sdr2.item(),sdr3.item(),sdr4.item()))
            
    print(f'SDR for source 1 (drums): {np.mean([x[0] for x in results])}')
    print(f'SDR for source 2 (bass): {np.mean([x[1] for x in results])}')
    print(f'SDR for source 3 (rest of accompaniment): {np.mean([x[2] for x in results])}')
    print(f'SDR for source 4 (vocals): {np.mean([x[3] for x in results])}')