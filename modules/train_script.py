
#imports
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
import argparse
import time
import pickle
from torchmetrics.audio import SignalNoiseRatio, ScaleInvariantSignalNoiseRatio
from model import ssnet
from sampler import Samples
from utils import utility_fct
from torch.nn import MSELoss


writer = SummaryWriter()
valid_writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('train', type=str)
parser.add_argument('valid', type=str)
parser.add_argument('batch_size', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('lr', type=float)
parser.add_argument('l2', type=float)
parser.add_argument('tag', type=str)
args = parser.parse_args()


if __name__ == "__main__":

    train = pickle.load(open(args.train, 'rb'))
    valid = pickle.load(open(args.valid, 'rb'))

    X_train, y_train = train
    X_valid, y_valid = valid


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    ssnet_ = ssnet().to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        ssnet_ = nn.DataParallel(ssnet_) 

    train_set = Samples(X_train, y_train)
    valid_set = Samples(X_valid, y_valid)

    train_loader = DataLoader(train_set, collate_fn=utility_fct, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valid_loader = DataLoader(valid_set, collate_fn=utility_fct, batch_size=args.batch_size, num_workers=8, shuffle=True)

    optimizer = optim.Adam(ssnet_.parameters(),lr=args.lr, weight_decay=args.l2)
    loss_function = MSELoss(reduction='mean').to(device)


    start_time = time.time()
    best_model = 1.0
    for epoch in range(args.epochs):
        ssnet_.train()
        train_losses = []
        for X, y in train_loader:
            size = len(X)
            X = X.view(size,2,-1).cuda()
            y = y.cuda(size,4,2,-1).cuda()
            out = ssnet_(X)[:,:,:,100:-100]
            ssnet_.zero_grad()
            loss = loss_function(out, y)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.cpu().detach().numpy())
        print(f'{epoch}_{np.mean(train_losses)}')
        
        ssnet_.eval()
        valid_losses = []
        for X, y in valid_loader:
            X = X.view(size,2,-1).cuda()
            y = y.view(size,4,2,-1).cuda()
            out = ssnet_(X)[:,:,:,100:-100]
            loss = loss_function(out, y)
            valid_writer.add_scalar("Loss/valid", loss, epoch)
            valid_losses.append(loss.cpu().detach().numpy())
        print(f'{epoch}_{np.mean(valid_losses)}')

        if np.mean(valid_losses) < best_model:
            best_model = np.mean(valid_losses)
            torch.save(ssnet_.state_dict(), f'ssnet_{args.tag}.pt')


    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time} seconds")

    writer.flush()
    valid_writer.flush()
    writer.close()
    valid_writer.close()