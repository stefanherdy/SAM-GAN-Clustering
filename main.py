import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.img_shape[1]*self.img_shape[2]),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(self.img_shape))), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Define the GAN model
class GAN(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.generator = Generator(self.latent_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)

    def forward(self, x):
        return self.generator(x)

# Define the custom Dataset class for the classified input images
class ClassifiedImagesDataset(Dataset):
    def __init__(self, images, labels, class_num):
        self.images = images
        self.labels = labels
        self.class_num = class_num

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if label == self.class_num:
            return image, label
        else:
            return None

# Function to generate artificial image of a certain class
def generate_image(gan, class_num, latent_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan.to(device)
    gan.eval()

    with torch.no_grad():
        noise = torch.randn(1, latent_dim).to(device)
        fake_image = gan.generator(noise)
    return fake_image

# Example usage
if __name__ == '__main__':
    # MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor()])
         #transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    latent_dim = 100  # latent dimension of the GAN
    img_shape = (1, 28, 28)  # image shape of the input images

    # Create a GAN model
    gan = GAN(latent_dim, img_shape)

    # Define loss functions
    adversarial_loss = nn.BCELoss()

    # Optimizers for generator and discriminator
    optimizer_G = optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(gan.discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))

    # Function to train the GAN
    def train_gan(dataloader, gan, optimizer_G, optimizer_D, adversarial_loss, epochs, device):
        gan.to(device)
        gan.train()

        for epoch in range(epochs):
            for i, (real_images, real_labels) in enumerate(dataloader):
                real_images = real_images.to(device)
                #inputs_train, real_labels = next(iter(dataloader))
                real_labels = torch.reshape(real_labels, (real_images.size(0),1)).type(torch.float32).to(device)
                # Train Discriminator
                optimizer_D.zero_grad()
                #real_labels = torch.ones(real_images.size(0), 1).to(device)
                fake_labels = real_labels
                #fake_labels = torch.zeros(real_images.size(0), 1).to(device)
                #fake_labels = torch.randint(0,9,(real_images.size(0), 1)).to(device).type(torch.float32)

                # Generate fake images
                #vorher torch.randn
                #z = torch.randint_like(0,9,(real_images.size(0), latent_dim)).to(device).type(torch.float32)
                z = torch.randn(real_images.size(0), latent_dim).to(device)
                #z = torch.randint(low=0, high, size
                fake_images = gan.generator(z)

                # Train discriminator on real images
                real_preds = gan.discriminator(real_images)
                real_loss = adversarial_loss(real_preds, real_labels)

                # Train discriminator on fake images
                fake_preds = gan.discriminator(fake_images.detach())
                fake_loss = adversarial_loss(fake_preds, fake_labels)

                # Total discriminator loss
                d_loss = real_loss# + fake_loss
                d_loss.backward()
                optimizer_D.step()

                # Train Generator
                optimizer_G.zero_grad()
                z = torch.randn(real_images.size(0), latent_dim).to(device)
                fake_images = gan.generator(z)
                fake_preds = gan.discriminator(fake_images)
                g_loss = adversarial_loss(fake_preds, real_labels)
                g_loss.backward()
                optimizer_G.step()

                # Print training progress
                print("[Epoch {}/{}] [Batch {}/{}] [D loss: {:.4f}] [G loss: {:.4f}]".format(
                    epoch+1, epochs, i+1, len(dataloader), d_loss.item(), g_loss.item()))

    # Train the GAN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_gan(trainloader, gan, optimizer_G, optimizer_D, adversarial_loss, epochs=1, device=device)

    # Generate artificial image of a certain class
    class_num = 7  # specify the class number of the desired artificial image
    fake_image = generate_image(gan, class_num, latent_dim)
    torchvision.utils.save_image(fake_image, 'fake_image_class_{}.png'.format(class_num))
    print("Generated artificial image of class {}: fake_image_class_{}.png saved!".format(class_num, class_num))
