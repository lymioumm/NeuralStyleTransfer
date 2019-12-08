import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

# 首先需要在项目文件夹下创建好output目录


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')


niter = 25
betal = 0.5
lr = 0.0002
outf = 'output'
img_size = 64
batch_size = 64

nz = 100  # size of latent vector
ngf = 64  # filter size of generator
ndf = 64  # filter size of discriminator
nc = 3    # output image channels


class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            #     input is Z,going into a convolution

            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            #     state size. (ngf*8)×4×4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            #     state size. (ngf*4)×8×8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            #     state size. (ngf*2)×16×16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #     state size. (ngf)×32×32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            #     state size.(nc)×64×64
        )

    def forward(self, input):
        output = self.main(input)
        return output
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.main = nn.Sequential(
            # input is (nc)×64×64
            nn.Conv2d(nc,ndf,4,2,1,bias = False),
            nn.LeakyReLU(0.2,inplace=True),
            # state size.(ndf)×32×32
            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),
            # state size.(ndf*2)×16×16
            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),
            # state size. (ndf*4)×8×8
            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),
            # state size. (ndf*8)×4×4
            nn.Conv2d(ndf*8,1,4,1,0,bias=False),
            nn.Sigmoid()
        )
    def forward(self,input):
        output = self.main(input)
        return output.view(-1,1).squeeze(1)


# weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)
def printNetDis():
    netD = Discriminator()
    netD.apply(weights_init)
    print(netD)
    pass

def printNetGen():
    netG = Generator()
    #  call the function on the network object
    netG.apply(weights_init)
    print(f'printNetGen:\n{netG}')

    pass

def printJupyNetGen():
    netG = _netG()
    netG.apply(weights_init)
    print(f'printJupyNetGen:\n{netG}')

    pass

def LoadingData():
    dataset = datasets.CIFAR10(root = 'data',download = True,
                              transform = transforms.Compose([
                                  transforms.Resize(img_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                              ]))
    dataloader = torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)
    return dataloader


    pass


def achieve():
    batch_size = 64
    netG = Generator()
    #  call the function on the network object
    # netG.apply(weights_init)
    netD = Discriminator()
    # netD.apply(weights_init)

    # Defining loss functions
    criterion = nn.BCELoss()
    input = torch.FloatTensor(batch_size,3,img_size,img_size)
    noise = torch.FloatTensor(batch_size,nz,1,1)
    fixed_noise = torch.FloatTensor(batch_size,nz,1,1).normal_(0,1)
    label = torch.FloatTensor(batch_size)
    real_label = 1
    fake_label = 0

    netD = netD.cuda()
    netG = netG.cuda()
    criterion = criterion.cuda()
    input,label = input.cuda(),label.cuda()
    noise,fixed_noise = noise.cuda(),fixed_noise.cuda()

    # Defining optimiser
    fixed_noise = Variable(fixed_noise)
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(),lr,betas = (betal,0.999))
    optimizerG = optim.Adam(netG.parameters(),lr,betas = (betal,0.999))



    # Training the complete network
    dataloader = LoadingData()

    for epoch in range(niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            if torch.cuda.is_available():
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
            labelv = Variable(label)

            output = netD(inputv)
            errD_real = criterion(output, labelv)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            labelv = Variable(label.fill_(fake_label))
            output = netD(fake.detach())
            errD_fake = criterion(output, labelv)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, labelv)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu, '%s/real_samples.png' % outf,normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % (outf, epoch),normalize=True)



    pass


def main():
    # printNetGen()
    # printJupyNetGen()
    # print("-----------")
    # print('\n\n\n')
    # printNetDis()
    achieve()

    pass


if __name__ == '__main__':
    main()

