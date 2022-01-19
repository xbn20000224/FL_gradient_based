#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy.matlib
from scipy import integrate
import math
import numpy as np

# def FedAvg(w):      # main中将w_locals赋给w，即worker计算出的权值
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():  # 对于每个参与的设备：
#         for i in range(1, len(w)):  # 对本地更新进行聚合
#             w_avg[k] += w[i][k]
#         w_avg[k] = torch.div(w_avg[k], len(w))
#     return w_avg


def fun1(x):
    return 1/x*np.exp(-x)

h_threshold = 0.1
integration = integrate.quad(fun1,h_threshold,np.inf)
# print(integration[0])

def FedAvg(w):

    w_avg = copy.deepcopy(w[0])   # 此处应拷贝梯度

    P0 = 1
    scale = (P0 / integration[0])**0.5
    # scale = 1.0
    for k in w_avg.keys():
        for i in range(1,len(w)):
            h = abs(float(np.random.randn(1,1)))
            # print(h)
            if h >= h_threshold:

                w_avg[k] += w[i][k]*scale

            else:
                w[i][k]=0
                w_avg[k] += w[i][k]



        w_avg[k] = torch.div(w_avg[k], len(w)*scale)


    return w_avg


# import copy
#
# origin = [1, 2, [3, 4]]
# cop1 = copy.copy(origin)
# cop2 = copy.deepcopy(origin)
# cop3=origin
#
# print(origin)
# print(cop1)
# print(cop2)
# print(cop3)
#
# origin[2][0] = "hey!"  #改变
# print("##################")
#
# print(origin)
# print(cop1)
# print(cop2)   # 深度拷贝不变
# print(cop3)
#
# [1, 2, [3, 4]]
# [1, 2, [3, 4]]
# [1, 2, [3, 4]]
# [1, 2, [3, 4]]
# ##################
# [1, 2, ['hey!', 4]]
# [1, 2, ['hey!', 4]]
# [1, 2, [3, 4]]
# [1, 2, ['hey!', 4]]
