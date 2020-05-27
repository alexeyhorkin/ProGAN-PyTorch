import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter



def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str,
                        default="dir/",
                        help="path to the logs files")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()

    title_size = 20
    loss_gen, loss_dis = [], []

    files = os.listdir(path=args.log_dir)
    files = [ i for i in files if '.log' in i ]
    files.sort()

    for file_ in files:
        with open(os.path.join(args.log_dir, file_), 'r') as log_file:  
            for line in log_file:
                _, d_loss, g_loss = list(map(float, line.split()))
                loss_dis.append(d_loss)
                loss_gen.append(g_loss)
    

    time = list(range(0, len(loss_gen)))

    window_size, pol_grade = 51, 3

    plt.title('Generator and Discriminator loss', fontsize=title_size)
    plt.plot(time ,savgol_filter(loss_dis, window_size, pol_grade), marker='o', markersize=3)
    plt.plot(time ,savgol_filter(loss_gen, window_size, pol_grade), marker='o', markersize=3)

    plt.xlabel('Number of batch', fontsize=title_size)
    plt.ylabel('Loss', fontsize=title_size)
    plt.legend(['Discriminator loss', 'Generator loss', 'tmp'], fontsize=title_size)
    plt.grid()
    plt.show()
                
