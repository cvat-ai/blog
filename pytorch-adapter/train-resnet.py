#!/usr/bin/env python

# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
This script trains ResNet-34 on the data from a CVAT task.
The resulting weights are saved as `weights.pth`.
"""

import argparse
import logging
import os

import cvat_sdk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms

from cvat_sdk.pytorch import TaskVisionDataset, ExtractSingleLabelIndex

def train_one_epoch(
    net: nn.Module, criterion: nn.Module,
    optimizer: optim.Optimizer, data_loader: torch.utils.data.DataLoader,
):
    WINDOW_SIZE = 100
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % WINDOW_SIZE == WINDOW_SIZE - 1:
            logging.info(f'[{i + 1}/{len(data_loader)}] loss: {running_loss / WINDOW_SIZE:.3f}')
            running_loss = 0.0


def run_training(
    dataset: torch.utils.data.Dataset,
    num_classes: int,
):
    BATCH_SIZE = 4
    NUM_EPOCHS = 2

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True)

    logging.info('Created data loader')

    net = models.resnet34(num_classes=num_classes)
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    logging.info('Started Training')

    for epoch in range(NUM_EPOCHS):
        logging.info(f'Starting epoch #{epoch}...')
        train_one_epoch(net, criterion, optimizer, data_loader)

    logging.info('Finished training')

    torch.save(net.state_dict(), "weights.pth")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_task_id', type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Starting...')

    with cvat_sdk.make_client(
        os.getenv("CVAT_HOST", 'app.cvat.ai'),
        credentials=(os.getenv("CVAT_USER"), os.getenv("CVAT_PASS")),
    ) as client:
        logging.info('Created the client')

        train_set = TaskVisionDataset(client, args.train_task_id,
            transform=transforms.Compose([
                transforms.RandomResizedCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]),
            target_transform=ExtractSingleLabelIndex(),
        )

        num_classes = len(client.tasks.retrieve(args.train_task_id).labels)

        logging.info('Created the training dataset')

        run_training(train_set, num_classes)

if __name__ == '__main__':
    main()
