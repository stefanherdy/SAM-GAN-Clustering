# -*- coding: utf-8 -*-

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

def main(name, perform_pretraining):
    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.set_deterministic(True) # Needed for reproducible results


    

    # Root directory for dataset
    if  perform_pretraining == False:
      dataroot = "./riccia_imgs_selected/" + name
      # Number of training epochs
      num_epochs = 20000
    else:
      dataroot = "./riccia_imgs_selected/"
      # Number of training epochs
      num_epochs = 5000
    # Number of workers for dataloader
    workers = 1

    # Batch size during training
    batch_size = 1

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 128


    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 128

    # Size of feature maps in discriminator+
    ndf = 32

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
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
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    # custom weights initialization called on ``netG`` and ``netD``
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
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 128 x 128
            )

        def forward(self, input):
            return self.main(input)

    # Create the generator
    if perform_pretraining == True:
      netG = Generator(ngpu).to(device)
    
      # Handle multi-GPU if desired
      if (device.type == 'cuda') and (ngpu > 1):
          netG = nn.DataParallel(netG, list(range(ngpu)))

      # Apply the ``weights_init`` function to randomly initialize all weights
      #  to ``mean=0``, ``stdev=0.02``.
      netG.apply(weights_init)
    else:
      netG = Generator(ngpu).to(device)

      # Handle multi-GPU if desired
      if (device.type == 'cuda') and (ngpu > 1):
          netG = nn.DataParallel(netG, list(range(ngpu)))


      netG.load_state_dict(torch.load('./models/netG_pretrained.pt'))
    # Print the model
    print(netG)


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 128 x 128
                nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False), 
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
                nn.Sigmoid()
                # state size. 1
            )

        def forward(self, input):
            return self.main(input)


    # Create the Discriminator
    if perform_pretraining == True:
      netD = Discriminator(ngpu).to(device)

      # Handle multi-GPU if desired
      if (device.type == 'cuda') and (ngpu > 1):
          netD = nn.DataParallel(netD, list(range(ngpu)))
        
      # Apply the ``weights_init`` function to randomly initialize all weights
      # like this: ``to mean=0, stdev=0.2``.
      netD.apply(weights_init)
    else:
      netD = Discriminator(ngpu).to(device)

      # Handle multi-GPU if desired
      if (device.type == 'cuda') and (ngpu > 1):
          netD = nn.DataParallel(netD, list(range(ngpu)))

      netD.load_state_dict(torch.load('./models/netD_pretrained.pt'))
    # Print the model
    print(netD)

    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        st = time.time()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)  
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            if (epoch % 500 == 0):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    if epoch < 0:
                        images = fake.detach().cpu().numpy()
                        dir_name = f'./fake_imgs/{name}/{name}'
                        os.makedirs(dir_name, exist_ok = True)
                        for k in range(images.shape[0]): 
                            img_name = f'img_{perform_pretraining}_{epoch}_{k}_{name}.png'
                            image_to_save = np.transpose(images[i,:,:,:])
                            image_to_save = (image_to_save-np.min(image_to_save))/(np.max(image_to_save)-np.min(image_to_save)) 
                            plt.imsave(os.path.join(dir_name, img_name), image_to_save)
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                real_batch = next(iter(dataloader))

                # Plot the real images
                plt.figure(figsize=(15,15))
                plt.subplot(1,2,1)
                plt.axis("off")
                plt.title("Real Images")
                plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

                # Plot the fake images from the last epoch
                plt.subplot(1,2,2)
                plt.axis("off")
                plt.title("Fake Images")
                plt.imshow(np.transpose(img_list[-1],(1,2,0)))
                #plt.show()
                plt.savefig(f'./species_plots/generated_imgs_{perform_pretraining}_{epoch}_{name}.png', dpi=300)
            iters += 1
        et = time.time()
        # get the execution time
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')

    # 
    # Save pretrained models
    if perform_pretraining == True:
      torch.save(netD.state_dict(), './models/netD_pretrained.pt')
      torch.save(netG.state_dict(), './models/netG_pretrained.pt')

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.suptitle(name)
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()
    
if __name__ == '__main__':
    names = ['bifurca_distal', 'bifurca_proximal', 'commutata_distal', 'commutata_proximal', 'crozalsii_distal', 'crozalsii_proximal', 'glauca_distal', 'glauca_proximal', 'gothica_distal', 'gothica_proximal', 'sorocarpa_distal', 'sorocarpa_proximal', 'warnsdorfii_distal', 'warnsdorfii_proximal']
    perform_pretraining = True
    if perform_pretraining == False:
      for name in names:
        main(name, perform_pretraining)
    else:
      main('all', perform_pretraining)
