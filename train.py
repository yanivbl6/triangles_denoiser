import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import sys

import wandb
from tqdm import tqdm
    



argparser = argparse.ArgumentParser()
argparser.add_argument('-d','--input_size', type=int, default=10, help='model name')
argparser.add_argument('-n','--hidden_size', type=int, default=1024, help='hidden size')


argparser.add_argument('-r','--range', type=float, default=10.0, help='set the range of the input data')
argparser.add_argument('-N','--num-samples', type=int, default=32, help='set the number of the input data centers')
argparser.add_argument('-a','--alpha', type=float, default=90, help='detenrmine the angle between close input data points')
argparser.add_argument('-s','--sigma', type=float, default=0.05, help='noise level')

argparser.add_argument('-S','--seed', type=int, default=0, help='set the random seed')
argparser.add_argument('-D','--device', type=int, default=0, help='set the device')

argparser.add_argument('-x','--dist', type=int, default=1.0, help='set the distance between the two neighbouring input data points')

argparser.add_argument('--lr', type=float, default=0.1, help='learning rate')
argparser.add_argument('--wd', type=float, default=0.1, help='weight decay')
argparser.add_argument('-b','--batch-size', type=int, default=128, help='batch size')
argparser.add_argument('-E','--epochs', type=int, default=1000, help='epochs')

argparser.add_argument('--misc', action='store_true', default=False, help='miscellaneous')


argparser.add_argument('--cosine', action='store_true', default=False, help='use cosine learning rate')

argparser.add_argument('-e','--predict-e', action='store_true', default=False, help='predict epsilon')


argparser.add_argument('--test-samples', type=int, default=1000, help='set the number of the test samples')


##define a single lyaer Fully connected network:
class FCN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def deg2rad(deg):
    return deg*np.pi/180.0


def create_datapoints(num_samples, input_size, mrange, deg, dist, device):
    alpha = deg2rad(deg)

    datapoints = (torch.rand(num_samples, input_size, device=device)-0.5)*(mrange*2)

    V1 = torch.randn(num_samples, input_size, device=device)
    V2 = torch.randn(num_samples, input_size, device=device)
    V2 = V2 - torch.sum(V2*V1, dim=1, keepdim=True)*V1/torch.sum(V1*V1, dim=1, keepdim=True)
    V1 = V1/torch.sqrt(torch.sum(V1*V1, dim=1, keepdim=True))
    V2 = V2/torch.sqrt(torch.sum(V2*V2, dim=1, keepdim=True))

    datapoints_2 = V1*dist
    datapoints_3 = args.dist*(V1*np.cos(alpha) + V2*np.sin(alpha))

    datapoints = torch.cat((datapoints, datapoints_2, datapoints_3), dim=0)

    ##shuffle the data points:
    datapoints = datapoints[torch.randperm(datapoints.shape[0])]
    return datapoints

def main(args):
    #set the random seed
    torch.manual_seed(args.seed)

    #set the device
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    
    #set the input data
    
    
    datapoints = create_datapoints(args.num_samples, args.input_size, args.range, args.alpha, args.dist , device)


    #set the model
    model = FCN(args.input_size, args.hidden_size).to(device)

    #set the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)

    #set the loss function
    loss_func = nn.MSELoss()

    #set the wandb
    if not args.misc:
        wandb.init(project="phased_denoiser", entity='dl-projects' ,  config=args)
        wandb.watch(model, log="all")


    ##create a data loader from the data points:

    dl = torch.utils.data.DataLoader(datapoints, batch_size=min(args.batch_size,datapoints.shape[0]), shuffle=True)



    #set the training loop
    bar = tqdm( range(args.epochs), total = args.epochs, desc="training")
    for epoch in bar:
        model.train()
        total_loss = 0     
        for x in dl:

            optimizer.zero_grad()
            
            noise = torch.randn_like(x)
            y = x + args.sigma* noise


            if args.predict_e:
                eat = model(y)
                loss = loss_func(eat, noise )
            else:
                xat = model(y)
                loss = loss_func(xat, x)


            loss.backward()
            optimizer.step()

            if args.cosine:
                scheduler.step()

            total_loss += loss.item()

        total_loss = total_loss/len(dl)        
        
        if wandb.run is not None:
            wandb.log({"epoch": epoch, "loss": total_loss, 'learning rate': optimizer.param_groups[0]['lr']})

        bar.set_postfix(loss=total_loss)


    if wandb.run is not None:
        wandb.log({"train_loss": total_loss})



    
    total_test_loss = 0  
    model.eval()

    iters=  args.test_samples//args.batch_size

    for _ in range(iters):
        
        x = (torch.rand(args.num_samples, args.input_size, device=device)-0.5)*(args.range*2)

        noise = torch.randn_like(x)
        y = x + args.sigma* noise

        if args.predict_e:
            eat = model(y)
            loss = loss_func(eat, noise )
        else:
            xat = model(y)
            loss = loss_func(xat, x)


        total_test_loss += loss.item()
    
    total_test_loss = total_test_loss/iters

    if wandb.run is not None:
        wandb.log({"test_loss": total_test_loss})
    print("test loss: ", total_test_loss)

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)