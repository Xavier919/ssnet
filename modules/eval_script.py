from scipy.io.wavfile import write
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
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
parser.add_argument('model')
args = parser.parse_args()

if __name__ == "__main__":

    test = pickle.load(open(args.test, 'rb'))
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

    ssnet_.eval()
    for X, y in test_loader:
        size = len(X)
        X = X.view(size,2,-1).cuda()
        y = y.view(size,2,-1).cuda()
        out = ssnet_(X)
        out = out.cpu().detach().numpy()


    #sample_rate = 44100  
    #wav_file = "output.wav"
    #write(wav_file, sample_rate, audio_data) 