""" Generate single ecg samples from a particular depth of a model """

import argparse
import torch as th
import numpy as np
import base64
import matplotlib.pyplot as plt
import os
from io import BytesIO
from torch.backends import cudnn
from pro_gan_pytorch.PRO_GAN import Generator
from tqdm import tqdm


def drawhtml(num_samples, signals, out_dir):
    resolution = signals[0].cpu().view(-1).shape[0]
    html = 'Seria of figures <br>'
    for i in tqdm(range(num_samples-1)):

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(signals[i].cpu().view(-1))

        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

        html += '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '<br>'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir,f'generated_samples_{resolution}.html'), 'w') as f:
        f.write(html)

# turn on the fast GPU processing mode on
cudnn.benchmark = True


# set the manual seed
th.manual_seed(8)

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator", required=True)

    parser.add_argument("--latent_size", action="store", type=int,
                        default=128,
                        help="latent size for the generator")

    parser.add_argument("--depth", action="store", type=int,
                        default=9,
                        help="depth of the network. **Starts from 1")

    parser.add_argument("--out_depth", action="store", type=int,
                        default=8,
                        help="output depth of images. **Starts from 0")

    parser.add_argument("--num_samples", action="store", type=int,
                        default=20,
                        help="number of synchronized grids to be generated")

    parser.add_argument("--out_dir", action="store", type=str,
                        default="ecg_samples/",
                        help="path to the output directory for the frames")

    args = parser.parse_args()

    return args



def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    print("Creating generator object ...")
    # create the generator object
    gen = th.nn.DataParallel(Generator(
        depth=args.depth,
        latent_size=args.latent_size
    ))

    print("Loading the generator weights from:", args.generator_file)
    # load the weights into it
    gen.load_state_dict(
        th.load(args.generator_file, map_location=str(device))
    )

    # path for saving the files:
    save_path = args.out_dir

    print("Generating scale synchronized images ...")
    # generate the images:
    with th.no_grad():
        point = th.randn(args.num_samples , args.latent_size)
        # point = (point / point.norm()) * (args.latent_size ** 0.5) # ?
        ecg_samples = gen(point, depth=args.out_depth, alpha=1)


    drawhtml(args.num_samples, ecg_samples, args.out_dir)


    print("Generated %d images at %s" % (args.num_samples, save_path))


if __name__ == '__main__':
    main(parse_arguments())