import argparse
import os
import sys
import json

import torch
from torch.utils.data import Dataset
from torch.backends import cudnn

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import datetime

from pro_gan_pytorch.PRO_GAN import ProGAN

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        for tr in self.transforms:
            x = tr(x)
        return x

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        signal = np.expand_dims(sample, 0) 
        return torch.from_numpy(signal).float() 

class Normalize(object):
    def __init__(self, mu=0, std=1):
        self.mu = mu
        self.std = std
    def __call__(self, x):
        return (x-self.mu)/self.std

class Ecg_dataset(Dataset):
    def __init__(self, size_of_data, data_path, transform=None):
        self.size_of_data = size_of_data
        self.data = json.load(open(os.path.join(data_path), 'r') )
        self.data_keys = list(self.data.keys()) 
        self.len = len(self.data_keys)
        self.otvedenie = 'i'
        self.transform = transform
        # self.statistics = get_statistics(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        key = self.data_keys[index]
        signal = np.array(self.data[key]['Leads'][self.otvedenie]['Signal'])
        size = len(signal.reshape(-1))
        start = rd.randint(0,size - self.size_of_data)
        res = signal[start:start+self.size_of_data]

        if self.transform:
            res = self.transform(res)
        
        return res


def get_statistics(data):
    mu = 0
    std = 0
    otvedenie = 'i'
    for i in data:
        mu += np.array(data[i]['Leads'][otvedenie]['Signal']).mean()
    mu /=len(data)
    for i in data:
        std += ((np.array(data[i]['Leads'][otvedenie]['Signal']) - mu)**2).mean()
    std = np.sqrt((std/len(data)))

    print(f'mu is {mu}, std is {std}')
    return mu, std

# turn on the fast GPU processing mode on
cudnn.benchmark = True

# define the device for the training script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main() -> None:

    exp_root = 'experiments'
    os.makedirs(exp_root, exist_ok=True)

    data_path = os.path.join('data', 'fix_data.json')

    # some parameters:
    size_of_data = 2048
    depth = 10
    # hyper-parameters per depth (resolution)
    num_epochs = [10, 20, 20, 60, 60, 80, 80, 100, 100, 100]
    fade_ins = [50]*depth
    batch_sizes = [20]*depth
    latent_size = 256
    lr = 0.0002

    # select the device to be used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the data. Ignore the test data and their classes
    transform = Compose([Normalize(mu=25.963, std=135.523), ToTensor()])
    dataset = Ecg_dataset(size_of_data, data_path, transform=transform)

    # ======================================================================
    # This line creates the PRO-GAN
    # ======================================================================
    pro_gan = ProGAN(depth=depth, 
                    latent_size=latent_size, loss='standard-gan', learning_rate=lr, device=device)
    # ======================================================================
    # This line trains the PRO-GAN
    # ======================================================================

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # ======================================================================
    # This is create info file with some useful information
    # ======================================================================
    writer = sys.stdout
    info_file_path = os.path.join(exp_root, current_time + '_exp_data')
    os.makedirs(info_file_path, exist_ok=True)
    with open(os.path.join(info_file_path, 'info.txt') , 'w') as f:
        sys.stdout = f
        print(pro_gan.gen)
        print(pro_gan.dis)
        print(f'size of data - {size_of_data}')
        print(f'num_epoches - {num_epochs}')
        print(f'fade_ins - {fade_ins}')
        print(f'batch sizes - {batch_sizes}')
        print(f'latent size - {latent_size}')
        print(f'depth - {depth}')
        print(f'learning rate - {lr}')
        sys.stdout = writer
    # ======================================================================

    # pro_gan.train(
    #     dataset=dataset,
    #     epochs=num_epochs,
    #     fade_in_percentage=fade_ins,
    #     batch_sizes=batch_sizes,
    #     feedback_factor=3,
    #     num_samples=9,
    #     log_dir=os.path.join(exp_root, current_time + '_exp_data' ),
    #     save_dir=os.path.join(exp_root, current_time + '_exp_data' ),
    #     sample_dir=os.path.join(exp_root, current_time + '_exp_data', 'samples')
    # )
    # ======================================================================  

    # testing and visualizate
    pretrained_gen = pro_gan.gen

    signal = dataset[2][0]
    plt.plot(signal)
    plt.show()

    noise = torch.randn(batch_sizes[0], latent_size).to(device)
    fake_data = pretrained_gen(noise, depth-1 , 1.0)
    signal = fake_data[0][0]
    signal = signal.detach().cpu().numpy()
    plt.plot(signal)
    plt.show()




if __name__ == '__main__':
    main()