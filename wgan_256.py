
import time
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(name, perform_pretraining, mode):
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) 

    if  perform_pretraining == False:
      dataroot = "./local/raw_imgs_cropped/" + mode + '/' + name
      num_epochs = 2001
    else:
      dataroot = "./local/raw_imgs_cropped/" + mode
      num_epochs = 5001

    workers = 8

    batch_size = 128

    image_size = 256


    # Number of channels 
    nc = 3

    # Size of z latent vector 
    nz = 200

    # Size of feature maps in generator
    ngf = 128

    # Size of feature maps in discriminator+
    ndf = 32

    # Learning rate 
    lr = 0.0002

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # Number of GPUs 
    ngpu = 1

    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(180, fill=255),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 16),
                nn.ReLU(True),
                # state size. (ngf*16) x 4 x 4
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 8 x 8
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 16 x 16 
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 32 x 32
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 64 x 64
                nn.ConvTranspose2d(ngf,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 128 x 128
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 256 x 256
            )

        def forward(self, input):
            return self.main(input)

    if perform_pretraining == True:
      netG = Generator(ngpu).to(device)
    
      if (device.type == 'cuda') and (ngpu > 1):
          netG = nn.DataParallel(netG, list(range(ngpu)))

      netG.apply(weights_init)
    else:
      netG = Generator(ngpu).to(device)

      if (device.type == 'cuda') and (ngpu > 1):
          netG = nn.DataParallel(netG, list(range(ngpu)))


      netG.load_state_dict(torch.load('./wgan/models/netG_pretrained_' + mode + '.pt'))
    print(netG)

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 256 x 256
                nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # input is (nc) x 128 x 128
                nn.Conv2d(ndf, ndf, 4, stride=2, padding=1, bias=False), 
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 64 x 64
                nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 32 x 32
                nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 16 x 16 
                nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 8 x 8
                nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*16) x 4 x 4
                nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),
                # No sigmoid in WGAN
                # nn.Sigmoid()
                # state size. 1
            )

        def forward(self, input):
            return self.main(input)


    if perform_pretraining == True:
      netD = Discriminator(ngpu).to(device)

      if (device.type == 'cuda') and (ngpu > 1):
          netD = nn.DataParallel(netD, list(range(ngpu)))
        
      netD.apply(weights_init)
    else:
      netD = Discriminator(ngpu).to(device)

      if (device.type == 'cuda') and (ngpu > 1):
          netD = nn.DataParallel(netD, list(range(ngpu)))

      netD.load_state_dict(torch.load('./wgan/models/netD_pretrained_' + mode + '.pt'))
    print(netD)

    def discriminator_loss(real_output, fake_output):
        return -torch.mean(real_output) + torch.mean(fake_output)

    def generator_loss(fake_output):
        return -torch.mean(fake_output)
    
    def clip_weights(netD, clip_value):
        for p in netD.parameters():
            p.data.clamp_(-clip_value, clip_value)
    
    def clip_weights(netD, clip_value):
        for p in netD.parameters():
            p.data.clamp_(-clip_value, clip_value)

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    top_images = []
    generated_images = []

    n_critic = 5  
    clip_value = 0.01  # Weight clipping range

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        st = time.time()
        for i, data in enumerate(dataloader, 0):
            for _ in range(n_critic):
                netD.zero_grad()
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)

                real_output = netD(real_cpu)
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
                fake_output = netD(fake.detach())

                errD = discriminator_loss(real_output, fake_output)
                errD.backward()
                optimizerD.step()

                clip_weights(netD, clip_value)

            netG.zero_grad()


            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            fake_output = netD(fake)

            errG = generator_loss(fake_output)
            errG.backward()
            optimizerG.step()

            if epoch > 1000 and epoch % 20 == 0:
                for j in range(b_size):
                    score = fake_output[j].item()
                    image = fake[j].detach().cpu().numpy()
                    if len(top_images) < 100:
                        top_images.append((score, image))
                    else:
                        min_score, _ = min(top_images, key=lambda x: x[0])
                        if score > min_score:
                            top_images.remove(min(top_images, key=lambda x: x[0]))
                            top_images.append((score, image))

            if epoch == 1400:
                for j in range(b_size):
                    score = fake_output[j].item()
                    image = fake[j].detach().cpu().numpy()
                    if len(generated_images) < 100:
                        generated_images.append((score, image))
                    else:
                        min_score, _ = min(generated_images, key=lambda x: x[0])
                        if score > min_score:
                            generated_images.remove(min(generated_images, key=lambda x: x[0]))
                            generated_images.append((score, image))

            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}")

            G_losses.append(errG.item())
            D_losses.append(errD.item())
                
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            if epoch > 1000:
                if epoch == num_epochs - 1:
                    dir_name = f'./wgan/most_real_imgs/{name}/'
                    os.makedirs(dir_name, exist_ok=True)
                    for idx, (_, img) in enumerate(sorted(top_images, key=lambda x: x[0], reverse=True)):
                        img_name = f'{idx}_{name}_{epoch}.png'
                        image_to_save = np.transpose(img, (1, 2, 0))
                        image_to_save = (image_to_save - np.min(image_to_save)) / (np.max(image_to_save) - np.min(image_to_save))
                        plt.imsave(os.path.join(dir_name, img_name), image_to_save)

            if epoch > 1000:
                if epoch == num_epochs - 1:
                    dir_name = f'./wgan/generated_imgs/{name}/'
                    os.makedirs(dir_name, exist_ok=True)
                    for idx, (_, img) in enumerate(sorted(generated_images, key=lambda x: x[0], reverse=True)):
                        img_name = f'{idx}_{name}_{epoch}.png'
                        image_to_save = np.transpose(img, (1, 2, 0))
                        image_to_save = (image_to_save - np.min(image_to_save)) / (np.max(image_to_save) - np.min(image_to_save))
                        plt.imsave(os.path.join(dir_name, img_name), image_to_save)

            if (epoch % 500 == 0):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    if epoch >= 0:
                        images = fake.detach().cpu().numpy()
                        dir_name = f'./wgan/fake_imgs_{perform_pretraining}/{name}/{name}'
                        os.makedirs(dir_name, exist_ok = True)
                        for k in range(64):  #images.shape[0]): 
                            img_name = f'img_{perform_pretraining}_{epoch}_{name}_{k}.png'
                            image_to_save = np.transpose(images[k,:,:,:])
                            image_to_save = (image_to_save-np.min(image_to_save))/(np.max(image_to_save)-np.min(image_to_save)) 
                            plt.imsave(os.path.join(dir_name, img_name), image_to_save)
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                real_batch = next(iter(dataloader))

                plt.figure(figsize=(15,15))
                plt.subplot(1,2,1)
                plt.axis("off")
                plt.title("Real Images")
                plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

                plt.subplot(1,2,2)
                plt.axis("off")
                plt.title("Fake Images")
                plt.imshow(np.transpose(img_list[-1],(1,2,0)))
                #plt.show()
                os.makedirs('./wgan/species_plots', exist_ok=True)
                plt.savefig('./wgan/species_plots/generated_imgs_' + name + '_' + str(epoch) + '.png', dpi=300)
                plt.close()
            iters += 1
        et = time.time()
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')

    os.makedirs('./wgan/models', exist_ok=True)
    if perform_pretraining == True:
      torch.save(netD.state_dict(), './wgan/models/netD_pretrained_' + mode + '.pt')
      torch.save(netG.state_dict(), './wgan/models/netG_pretrained_' + mode + '.pt')

    if perform_pretraining == False:
      torch.save(netD.state_dict(), './wgan/models/netD_pretrained_' + name + '_' + mode + '.pt')
      torch.save(netG.state_dict(), './wgan/models/netG_pretrained_' + name + '_' + mode + '.pt')


    real_batch = next(iter(dataloader))

    plt.figure(figsize=(15,15))
    plt.suptitle(name)
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))

if __name__ == '__main__':
    names = ['bifurca', 'commutata', 'crozalsii', 'glauca', 'gothica', 'sorocarpa', 'warnstorfii']
    modes = ['distal', 'proximal']

    for mode in modes:
        main(mode, True, mode)
    for mode in modes:
        for name in names:
            main(name + '_' + mode, False, mode)
