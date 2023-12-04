import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def main(args):
    # Root path
    if args.test == 'generated':
        root_data_path = './fake_imgs/'
    if args.test == 'norm':
        root_data_path = './_Sortierung/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((56, 56)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the dataset
    dataset = ImageFolder(root=root_data_path, transform=transform)

    num_total = len(dataset)
    num_train = int(args.train_val_ratio * num_total)
    num_val = num_total - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = resnet50(pretrained=True)
    num_classes = len(dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f'Epoch [{epoch+1}/{args.num_epochs}], Accuracy: {accuracy:.4f}')

        if (epoch + 1) % args.save_confusion_interval == 0:
            model.eval()
            all_labels = []
            all_predicted = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)

                    all_labels.extend(labels.cpu().numpy())
                    all_predicted.extend(predicted.cpu().numpy())

            cm = confusion_matrix(all_labels, all_predicted)

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(all_labels), yticklabels=np.unique(all_labels))
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            plt.savefig(f'confusion_matrix_plot_{args.test}_{epoch+1}.png')
            plt.close()
            
            # Save confusion matrix values
            np.savetxt(f'confmat_epoch_{args.test}_{epoch+1}.txt', cm, fmt='%d')
            print(f'Confusion matrix plot and values saved for epoch {epoch+1}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("AutoSeg")
    parser.add_argument("--learn_rate", type=int, default=0.001, help='Learn rate of optimizer')
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=5, help="Epochs between evaluation")
    parser.add_argument("--train_val_ratio", type=float, default=0.8, help="Ratio between train and evaluation data")
    parser.add_argument("--save_confusion_interval", type=int, default=5, help="Epochs between confusion matrix save")
    parser.add_argument("--test", choices=['norm', 'generated'], default='generated', help="Defines input folder, "
                        "'norm' = raw data, 'generated' = GAN-generated data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size")
    args = parser.parse_args()

    main(args)