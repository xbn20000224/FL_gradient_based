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


# def fun1(x):
#     return 1/x*np.exp(-x)
#
# h_threshold = 0.1
# integration = integrate.quad(fun1,h_threshold,np.inf)
# # print(integration[0])
#
# def FedAvg(w):
#
#     w_avg = copy.deepcopy(w[0])   # 此处应拷贝梯度
#
#     P0 = 1
#     scale = (P0 / integration[0])**0.5
#     # scale = 1.0
#     for k in w_avg.keys():
#         for i in range(1,len(w)):
#             h = abs(float(np.random.randn(1,1)))
#             # print(h)
#             if h >= h_threshold:
#
#                 w_avg[k] += w[i][k]*scale
#
#             else:
#                 w[i][k]=0
#                 w_avg[k] += w[i][k]
#
#
#
#         w_avg[k] = torch.div(w_avg[k], len(w)*scale)
#
#
#     return w_avg






# 对梯度进行聚合更新

# def fun1(x):
#     return 1/x*np.exp(-x)
#
# h_threshold = 0.1
# integration = integrate.quad(fun1,h_threshold,np.inf)
# # print(integration[0])
#
# def FedAvg(gradient):
#
#     grad_avg = copy.deepcopy(gradient[0])   # 此处应拷贝梯度
#
#     P0 = 1
#     scale = (P0 / integration[0])**0.5
#     # scale = 1.0
#     for k in grad_avg.keys():
#         for i in range(1,len(gradient)):
#             h = abs(float(np.random.randn(1,1)))
#             # print(h)
#             if h >= h_threshold:
#
#                 grad_avg[k] += gradient[i][k]*scale
#
#             else:
#                 gradient[i][k]=0
#                 grad_avg[k] += gradient[i][k]
#
#
#
#         grad_avg[k] = torch.div(grad_avg[k], len(gradient)*scale)
#
#
#
#     return grad_avg

g = [            {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':1, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':2, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':3, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':4, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':5, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':6, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':7, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':8},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0}]
# print(len(g))
'''
先试一个基础版本
'''
def FedAvg(gradient):      # main中将w_locals赋给w，即worker计算出的权值
    # grad_avg = copy.deepcopy(gradient[0])
    grad_avg = {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0}
    # print(grad_avg.keys())
    for k in grad_avg.keys():  # 对于每个参与的设备：
        for i in range(1, len(gradient)):  # 对本地更新进行聚合
            grad_avg[k] += gradient[i][k]
        grad_avg[k] = torch.div(grad_avg[k], len(gradient))
    return [grad_avg]

print(FedAvg(g))
