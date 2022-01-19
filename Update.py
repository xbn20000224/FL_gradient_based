#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

# 数据分组
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):

    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)


    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum) # 用常见的SGD优化器

        # ------------
        length = 8
        list_grad_bank = [(f"layer_{i}",0) for i in range(length)]
        grad_bank = dict(list_grad_bank)
        avg_counter = 0
        # ------------

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                # client_loss = loss

                loss.backward()

                # ------------------------------------------------
                for idx, param in enumerate(net.parameters()):
                    # print(enumerate(net.parameters()))
                    # print(f'{idx}: {param}')
                    grad_bank[f"layer_{idx}"] = grad_bank[f"layer_{idx}"] + param.grad.data
                    avg_counter += 1
                #-------------------------------------------------

                optimizer.step()


                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # ----------------------------------------------
        for key in grad_bank:
            grad_bank[key] = grad_bank[key] / avg_counter

        # print(grad_bank)
        # -----------------------------------------------

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), grad_bank

