#!/usr/bin/python3
# -*- coding: utf8 -*-

import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from net import SimpleConv3Net

data_dir = "./data/"
data_transforms = {
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(48),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val':
    transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}

data_loaders = {
    x: torch.utils.data.DataLoader(image_datasets[x],
                                   batch_size=16,
                                   shuffle=True,
                                   num_workers=4)
    for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


def train_model(model,
                criterion,
                optimizer,
                scheduler,
                num_epochs=100,
                use_gpu=True):
    for epoch in range(100):
        print("Epoch: {}/{}".format(epoch, num_epochs))
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_acc = 0.0

            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            for data in data_loaders[phase]:
                inputs, labels = data

                if use_gpu:
                    inputs = torch.autograd.Variable(inputs.cuda())
                    labels = torch.autograd.Variable(labels.cuda())

                else:
                    inputs = torch.autograd.Variable(inputs)
                    labels = torch.autograd.Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                running_loss += loss.data.item()
                running_acc += torch.sum(pred == labels).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc / dataset_sizes[phase]

            print("{}, Loss: {:.4f}, Acc: {:.4%}".format(
                phase, epoch_loss, epoch_acc))

    return model


if __name__ == "__main__":
    model = SimpleConv3Net()
    model.cuda()
    print(model)
    num_epochs = 200
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    use_gpu = torch.cuda.is_available()
    model = train_model(model, criterion, optimizer, scheduler, num_epochs,
                        use_gpu)
    save_dir = './models'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.ckpt'))
