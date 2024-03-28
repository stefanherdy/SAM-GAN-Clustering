#!/usr/bin/env python3

"""
Script Name: classify.py
Author: Stefan Herdy
Date: 07.12.2023
Description: 
Script to train a classification model and generate records of the model's performance.
Two different tests can be performed: one with real data and one with generated data.
The records can be used with read_records.py to compare the performance of the classification model with real and with generated data.

The script uses a pretrained ResNet50 model and the ImageFolder class from torchvision to load the data.
The model is trained with the Adam optimizer and the CrossEntropyLoss criterion.
The model's performance is evaluated with the accuracy and the confusion matrix.
The records are saved as json files and the confusion matrix is saved as a plot and as a txt file.

Usage: 
- To be able to run the "generated" test, you need to have generated fake images with the GAN. 
  Run the gan.py or the gan_128.py script to generate the fake images.
- Change the root_data_path to the path of the real or generated images.
- Run the script with the desired parameters.
"""

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
import os
import json

tests = ['norm', 'generated']


def main(args, test_num):
    # Root path
    if args.test == 'generated':
        root_data_path = './path/to/fake_imgs/'
    if args.test == 'norm':
        root_data_path = './path/to/real_imgs/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((56, 56)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root=root_data_path, transform=transform)

    num_total = len(dataset)
    num_train = int(args.train_val_ratio * num_total)
    num_val = num_total - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    print('Loading data...')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print('Loading model...')
    model = resnet50(pretrained=True)
    num_classes = len(dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)

    corrects = {}

    for epoch in range(args.num_epochs):
        model.train()
        print(f'Epoch {epoch+1}/{args.num_epochs}')
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
        
            corrects[epoch] = accuracy
        
            if not os.path.exists(args.records_path):
                print(f'Creating records folder at {args.records_path}')
                os.makedirs(args.records_path)
            
            print(f'Saving records of test {test_num} to {args.records_path}/validation_accuracy_{args.test}_{test_num}.json')
            json.dump(corrects, open(f'{args.records_path}/validation_accuracy_{args.test}_{test_num}.json', 'w'))


        if (epoch + 1) % args.save_confusion_interval == 0:
            model.eval()
            all_labels = []
            all_predicted = []
            all_outputs = []
            regression_matrix = np.zeros((num_classes, num_classes))

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)

                    all_labels.extend(labels.cpu().numpy())
                    all_predicted.extend(predicted.cpu().numpy())
                    all_outputs.extend(outputs.cpu().numpy())

            for k, output in enumerate(all_outputs):
                    regression_matrix[all_labels[k]] = regression_matrix[all_labels[k]] + output

            min_vals = np.min(regression_matrix, axis=1, keepdims=True)
            max_vals = np.max(regression_matrix, axis=1, keepdims=True)

            scaled_regression_matrix = (regression_matrix - min_vals) / (max_vals - min_vals)
            sum_vals = np.sum(scaled_regression_matrix, axis=1, keepdims=True)

            scaled_regression_matrix = scaled_regression_matrix / sum_vals

            cm = confusion_matrix(all_labels, all_predicted)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(all_labels), yticklabels=np.unique(all_labels))
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            plt.savefig(f'confusion_matrix_plot_{args.test}_{epoch+1}.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            sns.heatmap(scaled_regression_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=np.unique(all_labels), yticklabels=np.unique(all_labels))
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            plt.savefig(f'confusion_matrix_plot_{args.test}_{epoch+1}.png')
            plt.close()
            
            np.savetxt(f'confmat_epoch_{args.test}_{epoch+1}.txt', cm, fmt='%d')
            np.savetxt(f'regmat_epoch_{args.test}_{epoch+1}.txt', scaled_regression_matrix, fmt='%.2f')
            print(f'Confusion matrix plot and values saved for epoch {epoch+1}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser("AutoSeg")
    parser.add_argument("--learn_rate", type=int, default=0.001, help='Learn rate of optimizer')
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--train_val_ratio", type=float, default=0.8, help="Ratio between train and evaluation data")
    parser.add_argument("--save_confusion_interval", type=int, default=50, help="Epochs between confusion matrix save")
    parser.add_argument("--test", choices=tests, default='generated', help="Defines input folder, "
                        "'norm' = raw data, 'generated' = GAN-generated data")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Size")
    parser.add_argument("--num_tests", type=int, default=15, help="Number of tests to run")
    parser.add_argument("--records_path", type=str, default='./records', help="Path to records folder")

    args = parser.parse_args()

    
    for i in range(args.num_tests):
        print(f'Running test {i+1}/{args.num_tests}')
        main(args, i+1)