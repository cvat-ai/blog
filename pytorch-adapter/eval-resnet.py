#!/usr/bin/env python

# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
This script evaluates ResNet-34 on the data from a CVAT task.
The model weights are loaded from `weights.pth`.
"""

import argparse
import logging
import os

import cvat_sdk
import torch
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms

from torchmetrics.classification import MulticlassAccuracy

from cvat_sdk.pytorch import TaskVisionDataset, ExtractSingleLabelIndex


def run_evaluation(
    dataset: torch.utils.data.Dataset,
    num_classes: int,
):
    BATCH_SIZE = 4

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    logging.info('Created data loader')

    net = models.resnet34(num_classes=num_classes)
    net.load_state_dict(torch.load('weights.pth'))
    net.eval()

    logging.info('Started evaluation')

    metric = MulticlassAccuracy(num_classes=num_classes)

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for images, targets in data_loader:
            metric.update(net(images), targets)

    logging.info('Finished evaluation')

    print(f'Accuracy of the network: {metric.compute():.2%}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_task_id', type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Starting...')

    with cvat_sdk.make_client(
        os.getenv("CVAT_HOST", 'app.cvat.ai'),
        credentials=(os.getenv("CVAT_USER"), os.getenv("CVAT_PASS")),
    ) as client:
        logging.info('Created the client')

        test_set = TaskVisionDataset(client, args.test_task_id,
            transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]),
            target_transform=ExtractSingleLabelIndex()
        )

        num_classes = len(client.tasks.retrieve(args.test_task_id).labels)

        logging.info('Created the testing dataset')

        run_evaluation(test_set, num_classes)

if __name__ == '__main__':
    main()
