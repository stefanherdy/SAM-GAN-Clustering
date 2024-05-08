# -*- coding: utf-8 -*-

#%matplotlib inline
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
#from IPython.display import HTML

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# GAN with frozen weights of the discriminator model
# The pretrained discriminator model is frozen and the generator model is trained to generate images 
# that are classified as real by the discriminator model.
# This ensures that the generator model learns to generate images that are similar to the real images
# and represent the features of the real images.

def main(name, perform_pretraining, mode):
    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results


    ######################################################################
    # Inputs
    # ------
    # 
    # Let’s define some inputs for the run:
    # 
    # -  ``dataroot`` - the path to the root of the dataset folder. We will
    #    talk more about the dataset in the next section.
    # -  ``workers`` - the number of worker threads for loading the data with
    #    the ``DataLoader``.
    # -  ``batch_size`` - the batch size used in training. The DCGAN paper
    #    uses a batch size of 128.
    # -  ``image_size`` - the spatial size of the images used for training.
    #    This implementation defaults to 64x64. If another size is desired,
    #    the structures of D and G must be changed. See
    #    `here <https://github.com/pytorch/examples/issues/70>`__ for more
    #    details.
    # -  ``nc`` - number of color channels in the input images. For color
    #    images this is 3.
    # -  ``nz`` - length of latent vector.
    # -  ``ngf`` - relates to the depth of feature maps carried through the
    #    generator.
    # -  ``ndf`` - sets the depth of feature maps propagated through the
    #    discriminator.
    # -  ``num_epochs`` - number of training epochs to run. Training for
    #    longer will probably lead to better results but will also take much
    #    longer.
    # -  ``lr`` - learning rate for training. As described in the DCGAN paper,
    #    this number should be 0.0002.
    # -  ``beta1`` - beta1 hyperparameter for Adam optimizers. As described in
    #    paper, this number should be 0.5.
    # -  ``ngpu`` - number of GPUs available. If this is 0, code will run in
    #    CPU mode. If this number is greater than 0 it will run on that number
    #    of GPUs.
    #

    # Root directory for dataset
    #dataroot = "./use-segment-anything-model-to-autosegment-microscope-images/riccia_imgs_selected/bifurca_distal"
    if  perform_pretraining == False:
      dataroot = "./riccia_imgs_selected/" + mode + '/' + name
      # Number of training epochs
      num_epochs = 20
    else:
      dataroot = "./riccia_imgs_selected/" + mode
      # Number of training epochs
      num_epochs = 5001
    #use-segment-anything-model-to-autosegment-microscope-images/riccia_imgs_selected/bifurca_distal
    #dataroot = "./riccia_imgs_cropped"
    # Number of workers for dataloader
    workers = 8

    # Batch size during training
    batch_size = 128

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
    lr = 0.00001

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1


    ######################################################################
    # Data
    # ----
    # 
    # In this tutorial we will use the `Celeb-A Faces
    # dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`__ which can
    # be downloaded at the linked site, or in `Google
    # Drive <https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg>`__.
    # The dataset will download as a file named ``img_align_celeba.zip``. Once
    # downloaded, create a directory named ``celeba`` and extract the zip file
    # into that directory. Then, set the ``dataroot`` input for this notebook to
    # the ``celeba`` directory you just created. The resulting directory
    # structure should be:
    # 
    # ::
    # 
    #    /path/to/celeba
    #        -> img_align_celeba  
    #            -> 188242.jpg
    #            -> 173822.jpg
    #            -> 284702.jpg
    #            -> 537394.jpg
    #               ...
    # 
    # This is an important step because we will be using the ``ImageFolder``
    # dataset class, which requires there to be subdirectories in the
    # dataset root folder. Now, we can create the dataset, create the
    # dataloader, set the device to run on, and finally visualize some of the
    # training data.
    # 

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



    ######################################################################
    # Implementation
    # --------------
    # 
    # With our input parameters set and the dataset prepared, we can now get
    # into the implementation. We will start with the weight initialization
    # strategy, then talk about the generator, discriminator, loss functions,
    # and training loop in detail.
    # 
    # Weight Initialization
    # ~~~~~~~~~~~~~~~~~~~~~
    # 
    # From the DCGAN paper, the authors specify that all model weights shall
    # be randomly initialized from a Normal distribution with ``mean=0``,
    # ``stdev=0.02``. The ``weights_init`` function takes an initialized model as
    # input and reinitializes all convolutional, convolutional-transpose, and
    # batch normalization layers to meet this criteria. This function is
    # applied to the models immediately after initialization.
    # 

    # custom weights initialization called on ``netG`` and ``netD``
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    ######################################################################
    # Generator
    # ~~~~~~~~~
    # 
    # The generator, :math:`G`, is designed to map the latent space vector
    # (:math:`z`) to data-space. Since our data are images, converting
    # :math:`z` to data-space means ultimately creating a RGB image with the
    # same size as the training images (i.e. 3x64x64). In practice, this is
    # accomplished through a series of strided two dimensional convolutional
    # transpose layers, each paired with a 2d batch norm layer and a relu
    # activation. The output of the generator is fed through a tanh function
    # to return it to the input data range of :math:`[-1,1]`. It is worth
    # noting the existence of the batch norm functions after the
    # conv-transpose layers, as this is a critical contribution of the DCGAN
    # paper. These layers help with the flow of gradients during training. An
    # image of the generator from the DCGAN paper is shown below.
    #
    # .. figure:: /_static/img/dcgan_generator.png
    #    :alt: dcgan_generator
    #
    # Notice, how the inputs we set in the input section (``nz``, ``ngf``, and
    # ``nc``) influence the generator architecture in code. ``nz`` is the length
    # of the z input vector, ``ngf`` relates to the size of the feature maps
    # that are propagated through the generator, and ``nc`` is the number of
    # channels in the output image (set to 3 for RGB images). Below is the
    # code for the generator.
    # 

    # Generator Code

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


    ######################################################################
    # Now, we can instantiate the generator and apply the ``weights_init``
    # function. Check out the printed model to see how the generator object is
    # structured.
    # 

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


      netG.load_state_dict(torch.load('./models/netG_pretrained_' + name + '_' + mode + '.pt'))
    # Print the model
    print(netG)


    ######################################################################
    # Discriminator
    # ~~~~~~~~~~~~~
    # 
    # As mentioned, the discriminator, :math:`D`, is a binary classification
    # network that takes an image as input and outputs a scalar probability
    # that the input image is real (as opposed to fake). Here, :math:`D` takes
    # a 3x64x64 input image, processes it through a series of Conv2d,
    # BatchNorm2d, and LeakyReLU layers, and outputs the final probability
    # through a Sigmoid activation function. This architecture can be extended
    # with more layers if necessary for the problem, but there is significance
    # to the use of the strided convolution, BatchNorm, and LeakyReLUs. The
    # DCGAN paper mentions it is a good practice to use strided convolution
    # rather than pooling to downsample because it lets the network learn its
    # own pooling function. Also batch norm and leaky relu functions promote
    # healthy gradient flow which is critical for the learning process of both
    # :math:`G` and :math:`D`.
    # 

    #########################################################################
    # Discriminator Code

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


    ######################################################################
    # Now, as with the generator, we can create the discriminator, apply the
    # ``weights_init`` function, and print the model’s structure.
    # 

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

      netD.load_state_dict(torch.load('./models/netD_pretrained_' + name + '_' + mode + '.pt'))

    # Print the model
    print(netD)


    ######################################################################
    # Loss Functions and Optimizers

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


    ######################################################################
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    #noise_init = torch.randn(batch_size, nz, 1, 1, device=device)
    random_tensor = torch.randn(nz, 1, 1, device=device)

    # Expand the tensor to have size 128 x nz x 1 x 1
    noise_init = random_tensor.expand(128, -1, -1, -1)

    for epoch in range(num_epochs):
        st = time.time()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
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
            #errD_real.backward()
            D_x = output.mean().item()
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = noise_init[:b_size,:,:,:]
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            #errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            # optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
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
            if (iters % 2 == 0):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    if epoch > 0:
                        images = fake.detach().cpu().numpy()
                        dir_name = f'./fake_imgs_frozen/{name}/{name}'
                        os.makedirs(dir_name, exist_ok = True)
                        for k in range(images.shape[0]):
                            if k < 5:
                                img_name = f'img_{perform_pretraining}_{k}_{epoch}_{name}.png'
                                image_to_save = np.transpose(images[k,:,:,:])
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
                plt.savefig('./species_plots_v2/generated_imgs_' + name + '_' + str(epoch) + '.png', dpi=300)
            iters += 1
        et = time.time()
        # get the execution time
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')

    ######################################################################
    # Save pretrained models
    if perform_pretraining == True:
      torch.save(netD.state_dict(), './models/netD_pretrained_' + mode + '.pt')
      torch.save(netG.state_dict(), './models/netG_pretrained_' + mode + '.pt')


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
    #plt.show()
    #plt.savefig('generated_imgs_10000_bifurca_distal.png', dpi=300)



if __name__ == '__main__':
    names = ['bifurca', 'commutata', 'crozalsii', 'glauca', 'gothica', 'sorocarpa', 'warnstorfii']

    #perform_pretraining = True
    #modes = ['distal', 'proximal']
    #for mode in modes:
    #  if perform_pretraining == False:
    #    for name in names:
    #      main(name + '_' + mode, perform_pretraining, mode)
    #  else:
    #    main(mode, perform_pretraining, mode)

    perform_pretraining = False
    modes = ['distal', 'proximal']
    for mode in modes:
      if perform_pretraining == False:
        for name in names:
          main(name + '_' + mode, perform_pretraining, mode)
      else:
        main(mode, perform_pretraining, mode)
