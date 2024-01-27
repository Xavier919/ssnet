
#imports
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import pickle
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
import argparse
import time
from modules.model import ssnet
from modules.sampler import Samples
from modules.utils import utility_fct, get_loss


writer = SummaryWriter()
valid_writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('train_split')
parser.add_argument('valid_split')
parser.add_argument('batch_size', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('lr', type=float)
parser.add_argument('l2', type=float)
parser.add_argument('tag', type=str)
args = parser.parse_args()


if __name__ == "__main__":

    X_train, y_train = args.train_split
    X_valid, y_valid = args.valid_split


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    ssnet_ = ssnet().float().to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        ssnet_ = nn.DataParallel(ssnet_) 

    train_set = Samples(X_train, y_train)
    valid_set = Samples(X_valid, y_valid)

    train_loader = DataLoader(train_set, collate_fn=utility_fct, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_set, collate_fn=utility_fct, batch_size=args.batch_size)

    optimizer = optim.Adam(ssnet_.parameters(),lr=args.lr, weight_decay=args.l2)
    loss_function = nn.MSELoss(reduction='mean').to(device)

    start_time = time.time()

    best_model = 1.0
    for epoch in range(args.epochs):
        ssnet_.train()
        train_losses = []
        for X, y in train_loader:
            size = len(X)
            X = X.view(size,2,-1).cuda()
            y = y.view(size,2,-1).cuda()
            out = ssnet_(X)
            ssnet_.zero_grad()
            loss = get_loss(X, y, out, loss_function)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss)
        print(np.mean(train_losses))
        
        ssnet_.eval()
        valid_losses = []
        for X, y in valid_loader:
            size = len(X)
            X = X.view(size,2,-1).cuda()
            y = y.view(size,2,-1).cuda()
            out = ssnet_(X)
            loss = get_loss(X, y, out, loss_function)
            valid_writer.add_scalar("Loss/valid", loss, epoch)
            valid_losses.append(loss)
        print(np.mean(valid_losses))

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