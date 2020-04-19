import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from loggers import Loggers
from load_data import load_data
from train import train_gen, train_disc
from time import time, gmtime, strftime
from model import Generator, Discriminator
from utility import create_gif, save_results, save_checkpoint, plot


def train_gan(gen, disc, gen_opt, disc_opt, criterion, device, epoch, train_loader):
    gen_loss = []
    disc_loss = []

    epoch_start = time()
    for img, label in train_loader:
        disc_loss.append(train_disc(disc, gen, disc_opt, criterion, img, device))
        gen_loss.append(train_gen(gen, disc, gen_opt, criterion, img, device))

    end = time() - epoch_start
    Loggers.info('epoch = {} | disc_loss = {} | gen_loss = {} | time elapsed in epoch = {}'.format(epoch + 1,
                                                                                                   torch.mean(torch.FloatTensor(disc_loss)),
                                                                                                   torch.mean(torch.FloatTensor(gen_loss)),
                                                                                                   strftime("%H:%M:%S", gmtime(end))))
    return gen_loss, disc_loss


def main():
    torch.manual_seed(1)
    device = torch.device('cuda')
    mode = input('Enter 1 to train or 2 to view results\n')
    total_loss = {'disc_loss': [], 'gen_loss': []}

    if mode == '1':
        num_epochs = 100
        batch_size = 128
        lr = 0.0002
        gen = Generator(ip=100, op=28 * 28).to(device)
        disc = Discriminator(ip=28 * 28, op=1).to(device)
        gen_opt = optim.Adam(gen.parameters(), lr=lr)
        disc_opt = optim.Adam(disc.parameters(), lr=lr)
        criterion = nn.BCELoss()
        train_loader, test_loader = load_data(batch_size=batch_size)

        start = time()

        for epoch in range(num_epochs):
            gen_loss, disc_loss = train_gan(gen, disc, gen_opt, disc_opt, criterion, device, epoch, train_loader)
            total_loss['gen_loss'].append(torch.mean(torch.FloatTensor(gen_loss)))
            total_loss['disc_loss'].append(torch.mean(torch.FloatTensor(disc_loss)))

            save_results(epoch, gen=gen, show=False, save=True, path='result/output/output_{}.png', device='cuda')
        save_checkpoint(gen=gen, disc=disc)

        end = time() - start
        Loggers.info('Training done | Time Elapsed --> {}'.format(strftime("%H:%M:%S", gmtime(end))))

        # saving losses
        with open('result/losses.pkl', 'wb') as f:
            pickle.dump(total_loss, f)

        create_gif(path='result/output')  # create gif of output images

    elif mode == '2':
        plot(show=True, save=True, path='result/loss_plot.png')

    else:
        main()


if __name__ == '__main__':
    Loggers.info('Training on {}'.format(torch.cuda.get_device_name(0)))
    main()
