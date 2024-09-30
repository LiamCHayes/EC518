import time
import numpy as np
import random
import argparse

import torch

from network import DaggerNetwork 
from dataset import get_dataloader
import matplotlib.pyplot as plt

def train(data_folder, save_path):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    infer_action = DaggerNetwork()
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-4)
    gpu = torch.device('cuda')

    nr_epochs = 10
    batch_size = 32
    start_time = time.time()

    train_loader = get_dataloader(data_folder, batch_size)
    test_loader = get_dataloader(data_folder+'test/', batch_size)

    loss_per_epoch = np.empty(nr_epochs)
    valid_loss = np.empty(nr_epochs)
    for epoch in range(nr_epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []

        for batch_idx, batch in enumerate(train_loader):
            batch_in, batch_gt = batch[0].to(gpu), batch[1].to(gpu)

            batch_out = infer_action(batch_in)
            loss = mae(batch_out, batch_gt)
            if epoch == 10 and batch_idx == 0:
                print(batch_gt[:5])
                print(batch_out[:5])
                input('press enter...')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))
        
        # Calculate losses for train and validation
        loss_per_epoch[epoch] = total_loss
        for valid_batch_idx, valid_batch in enumerate(test_loader):
            valid_batch_in, valid_batch_gt = valid_batch[0].to(gpu), valid_batch[1].to(gpu)

            valid_batch_out = infer_action(valid_batch_in)
            loss = mae(valid_batch_out, valid_batch_gt)
            valid_loss[epoch] += loss
        
        print("\t\t Test loss: ", valid_loss[epoch])
        
    torch.save(infer_action, save_path)
    
    # Plot loss
    fig, ax = plt.subplots()
    ax.plot(range(nr_epochs), loss_per_epoch, color='r')
    ax.plot(range(nr_epochs), valid_loss, color='g')
    plt.savefig('./output/dagger02.png')
    plt.show()


def mae(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
                    C = number of classes
    batch_out:      torch.Tensor of size (batch_size, C) Predicted values
    batch_gt:       torch.Tensor of size (batch_size, C) True values
    return          float
    """
    loss = torch.sum(torch.abs(batch_out - batch_gt))
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC518 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="./", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="./", type=str, help='path where to save your model in .pth format')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)
