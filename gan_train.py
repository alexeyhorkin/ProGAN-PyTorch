import torch
import numpy as np
import random as rd
from torch.utils.data import Dataset, DataLoader
from pro_gan_pytorch.PRO_GAN import ProGAN
import matplotlib.pyplot as plt

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        array = sample 
        return torch.from_numpy(array).float() 

class Sin_dataset(Dataset):

    def __init__(self, size, transform=None):
        ''' 
        Class for custom dataset for ecg signals
       '''
        self.transform = transform
        self.size = size

    def __len__(self):
            return 100
    
    def __getitem__(self, index):
        t = np.linspace(0,1.5,self.size)
        # print(t)
        c = rd.random()*10
        sample = np.sin(c*t)
        # print(sample)
        sample = np.expand_dims(sample, axis=0)


        if self.transform:
            sample = self.transform(sample)
        
        return sample


if __name__ == '__main__':

    # some parameters:
    size_of_data = 64
    depth = 5
    # hyper-parameters per depth (resolution)
    num_epochs = [4]*depth
    fade_ins = [50]*depth
    batch_sizes = [20]*depth
    latent_size = 16

    # select the device to be used for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the data. Ignore the test data and their classes
    dataset = Sin_dataset(size_of_data, ToTensor())

    # ======================================================================
    # This line creates the PRO-GAN
    # ======================================================================
    pro_gan = ProGAN(depth=depth, 
                    latent_size=latent_size, device=device)
    # ======================================================================

    # ======================================================================
    # This line trains the PRO-GAN
    # ======================================================================
    pro_gan.train(
        dataset=dataset,
        epochs=num_epochs,
        fade_in_percentage=fade_ins,
        batch_sizes=batch_sizes,
        feedback_factor=10
    )
    # ======================================================================  

    # testing and visualizate
    pretrained_gen = pro_gan.gen

    signal = dataset[0][0]
    print(signal)
    plt.plot(signal)
    # plt.show()

    noise = torch.randn(batch_sizes[0], latent_size).to(device)
    fake_data = pretrained_gen(noise, depth-1 , 1.0)
    signal = fake_data[0][0]
    signal = signal.detach().cpu().numpy()
    plt.plot(signal)
    plt.show()

