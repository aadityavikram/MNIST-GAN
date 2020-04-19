import torch
from torch.autograd import Variable


def train_gen(gen, disc, optimizer, criterion, x, device):
    gen.zero_grad()

    x = Variable(x.view(-1, 28 * 28).to(device))
    mini_batch = x.size()[0]
    z = Variable(torch.randn((mini_batch, 100)).to(device))
    y = Variable(torch.ones(mini_batch).to(device))

    # forward pass
    gen_op = gen(z)
    disc_op = disc(gen_op)
    gen_loss = criterion(disc_op, y)

    # backward pass optimizing only gen's param
    gen_loss.backward()
    optimizer.step()

    return gen_loss.item()


def train_disc(disc, gen, optimizer, criterion, x, device):
    disc.zero_grad()

    # training on real data
    x = Variable(x.view(-1, 28 * 28).to(device))
    mini_batch = x.size()[0]
    y_real = Variable(torch.ones(mini_batch).to(device))

    disc_op = disc(x)
    disc_real_loss = criterion(disc_op, y_real)

    # training on fake data
    z = Variable(torch.randn(mini_batch, 100).to(device))
    gen_op = gen(z)
    y_fake = Variable(torch.zeros(mini_batch).to(device))

    disc_op = disc(gen_op)
    disc_fake_loss = criterion(disc_op, y_fake)

    # backward pass optimizing only disc's param
    disc_loss = disc_real_loss + disc_fake_loss
    disc_loss.backward()
    optimizer.step()

    return disc_loss.item()
