import argparse
import pickle
import torch as th
import numpy as np
import base64
import matplotlib.pyplot as plt
import os
from io import BytesIO
from torch.backends import cudnn
from pro_gan_pytorch.PRO_GAN import Generator
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

# turn on the fast GPU processing mode on
cudnn.benchmark = True


# set the manual seed
th.manual_seed(6)

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

    parser.add_argument("--count_iterations", action="store", type=int,
                        default=100,
                        help="dozadaysy")

    parser.add_argument("--out_dir", action="store", type=str,
                        default="ecg_interpolations/",
                        help="path to the output directory for the frames")

    args = parser.parse_args()

    return args

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

def get_batch(args):
    res = []
    a = th.randn(args.latent_size)
    b = th.randn(args.latent_size)
    alpha_arr = np.linspace(1,0,args.count_iterations)
    for alpha in alpha_arr:
        tmp = (alpha*a + (1-alpha)*b).cpu().numpy()
        res.append(tmp)
    res = np.array(res)

    return th.from_numpy(res)

def visualize_latent_space(batch, args):
    count_of_step , interpol_arr = args.count_iterations-1, []
    folder_path_to_save = args.out_dir
    postfix_for_file, title_size = '_ecg', 20
    final_out = batch
    fig = plt.figure(3, figsize=(16,9))
    ax1 = fig.add_subplot(1, 1, 1)
    def animate(i):
        y = final_out[i]
        ax1.clear()
        # ax1.plot(start_true)
        # ax1.plot(end_true)
        ax1.plot(y.cpu().numpy()[0], color = 'k')
        ax1.legend(['latent point'], loc='upper left', fontsize=title_size)
        interpol_arr.append(y.cpu().numpy()[0])
        plt.xlabel('time', fontsize=title_size)
        plt.ylabel('signal', fontsize=title_size)
        plt.title("iteration "+ str(i)+ "/"+ str(count_of_step+1), fontsize=title_size)
    anim = FuncAnimation(fig, animate,frames=count_of_step+1)

    if not os.path.exists(os.path.dirname(folder_path_to_save)): # create folder if folder doesn't exist
        os.makedirs(folder_path_to_save)
    save_path  = os.path.join(folder_path_to_save, 'animation_interpol' + postfix_for_file + '.gif')
    anim.save(save_path, writer='imagemagick', fps=60)
    # plt.show()

    save_path = os.path.join(folder_path_to_save, 'interpol_array'+ postfix_for_file +'.pickle')
    with open(save_path, 'wb') as q:
        # pickle.dump([start_true, end_true, interpol_arr], q)
        pickle.dump([interpol_arr], q)



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

    print("Generating scale synchronized images ...")
    # generate the images:
    with th.no_grad():
        batch = get_batch(args).to(device)
        ecg_samples = gen(batch, depth=args.out_depth, alpha=1)


    # drawhtml(args.count_iterations, ecg_samples, args.out_dir)
    visualize_latent_space(ecg_samples, args)


    print("Generated %d images at %s" % (args.count_iterations, args.out_dir))

if __name__ == '__main__':
    main(parse_arguments())
