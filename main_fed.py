#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# 下面是自带库
import matplotlib
matplotlib.use('Agg')                                             # 可理解为令pycharm不产生图片
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

# 下面是自建库
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid  # 引入了三种 iid 与 non-iid 数据库类型
from utils.options import args_parser  # 导入程序运行对应选项，在终端cd到对应文件夹即可以所需方式运行程序，如：python main_fed.py --dataset mnist  --num_channels 1 --model cnn --epochs 50 --gpu -1
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


# ------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # parse args 解析参数
    args = args_parser()
    # 将torch.Tensor()进行分配到设备，来选择cpu还是gpu进行训练
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users 加载数据集并进行用户分组
    if args.dataset == 'mnist':   # 如果用mnist数据集
        # 数据集的均值与方差
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)  # 训练集
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)  # 测试集
        # sample users
        if args.iid:  # 如果是iid的mnist数据集
            dict_users = mnist_iid(dataset_train, args.num_users)  # 为用户分配iid数据
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)  # 否则为用户分配non-iid数据
    # elif args.dataset == 'cifar':  # 如果用cifar数据集
    #     # 数据集均值与方差
    #     trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #     dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    #     dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    #     if args.iid:  # 如果是iid的cifar数据集
    #         dict_users = cifar_iid(dataset_train, args.num_users)  # 为用户分配iid数据
    #     else:  # cifar未设置non-iid数据
    #         exit('Error: only consider IID setting in CIFAR10')
    # else:
    #     exit('Error: unrecognized dataset')

    # 设置图像的size
    img_size = dataset_train[0][0].shape

    #---------------------------------------------------------------------------------------------------------------------------------------

    # build model 选择模型
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)

    elif args.model == 'cnn' and args.dataset == 'mnist':

        # 训练mnist的网络
        net_glob = CNNMnist(args=args).to(args.device)

    # elif args.model == 'mlp':
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #     net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    # else:
    #     exit('Error: unrecognized model')



    print(net_glob) #PS使用的全局模型

    net_glob.train()    # 训练一次

    '''
    此处应传梯度而非权重 即传 w = w - β△w 中的△w
    '''
    # copy weight
    w_glob = net_glob.state_dict()


    '''
    定义 所有worker聚合（fedavg()）得到的PS使用的梯度 grad_glob ，PS利用 grad_glob 进行GD
    Q：怎样对grad_glob初始化？
    '''
    grad_glob = [{'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0},
                 {'layer_0':0, 'layer_1':0, 'layer_2':0, 'layer_3':0, 'layer_4':0, 'layer_5':0, 'layer_6':0, 'layer_7':0}]   # 怎样对其初始化？？？


    # training_initialize
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:   # 选择利用所有用户进行聚合
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]  # 每一个local的w与全局w相等
        gradient_locals = [grad_glob for i in range(args.num_users)]

    for iter in range(args.epochs):
        loss_locals = []  # 对于每一个epoch，初始化worker的损失

        # if not args.all_clients:  # 如果不是用所有用户进行聚合
        #     w_locals = []  # 此时worker的w与全局w并不一致
        #
        #     gradient_locals = []  # 本地更新的模型的gradient


        m = max(int(args.frac * args.num_users), 1)  # 此时，在每一轮中，在所有worker中选取C-fraction（C∈（0,1））部分进行训练，m为选取的worker总数
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # 在所有的worker中(0,1,2...num_workers-1)选取m个worker（m = all_workers * C_fraction）,且输出不重复

        for idx in idxs_users:  # 对于选取的m个worker

            #----------
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  # 导入该worker的训练模型，对该worker进行本地训练更新
            # 此处是在PS处进行模型训练，得到全局的权值、梯度、损失
            w, loss, gradient = local.train(net=copy.deepcopy(net_glob).to(args.device))  # gradient 是保存梯度的字典
            #----------

            if args.all_clients:  # 如果选择全部用户进行聚合，则将本地更新的权值 赋给 该用户

                w_locals[idx] = copy.deepcopy(w)  # 对于每个worker，用其本地训练的权值进行更新

                gradient_locals[idx] = copy.deepcopy(gradient)  # 对于每个worker，用其本地训练的梯度进行更新


            # else:  # 并行
            #     w_locals.append(copy.deepcopy(w))
            #     gradient_locals.append(copy.deepcopy(gradient))


            loss_locals.append(copy.deepcopy(loss))

        # update and store global weights && gradients




        # print(gradient_locals[0]['layer_1'])

        '''
        利用FedAvg()函数对梯度进行聚合更新，并将结果赋给PS，PS利用聚合得到的梯度进行GD来更新权值
        Q:咋用grad_glob进行GD
        '''

        '''
        伪代码，PS利用聚合得到的梯度进行GD
        # grad_glob = FedAvg(gradient_locals)
        # net_glob.GD(grad_glob)  
        '''

        # print(gradient_locals)


        '''
        # INITIAL VERSION
        w_glob = FedAvg(w_locals)  # 利用选取的局部w对全局w进行聚合更新，w_glob即为全局聚合更新后的值
        print(w_glob)
        net_glob.load_state_dict(w_glob)  # copy weight to net_glob
        '''
        # print(np.size(w_locals),np.size(gradient_locals))
        # print(np.shape(w_locals),np.shape(gradient_locals))
        # print(w_locals)
        # print(gradient_locals)
        # print(np.shape(gradient_locals))

        '''
        尝试GD——start
        '''

        grad_glob = FedAvg(gradient_locals)

        # print(grad_glob)
        # print(np.size(w_locals),np.shape(w_locals),w_locals)
        for ele in w_locals:
            ele = list(ele)
            for ele1 in ele:
                print(type(ele1),ele1)


        '''
        定位在 Fed对于 gradient_locals操作后出错--更改Fed.py中的算法写法，修改后元素索引出错，无论正序倒序只能提取文字list，无法提取tensor数据
        '''

        # lr = 0.01
        # count_outercycle = 0
        # count_innercycle = 0
        # for ele in w_locals:
        #     ele = list(ele)
        #     for ele1 in ele:
        #         ele1 = list(ele1)
        #         # print(ele1)
        #         # print(grad_glob[count_outercycle][f"layer_{count_innercycle}"])
        #         ele1[1] += grad_glob[count_outercycle][f"layer_{count_innercycle}"] * lr
        #
        #         count_innercycle += 1
        #
        #     count_outercycle += 1
        #     count_innercycle = 0
        #
        # w_glob = w_locals
        # net_glob.load_state_dict(w_glob)

        lr = 0.1
        count_outercycle = 0
        count_innercycle = 0
        for ele in w_locals:
            ele = list(ele)
            print(ele)
            for ele1 in ele :
                ele1 = list(ele1)
                print(ele1)
                # print(grad_glob[count_outercycle][f"layer_{count_innercycle}"])
                ele1[-1] += grad_glob[0][f"layer_{count_innercycle}"]

                count_innercycle += 1

            count_outercycle += 1
            count_innercycle = 0

        w_locals_update=[]
        for ele2 in w_locals:
            n=tuple(ele2)
            w_locals_update.append((n))
            # print(l1)
            # print(l)

        w_glob = w_locals_update
        print(w_glob)
        net_glob.load_state_dict(w_glob)

        '''
        尝试GD--end
        '''









        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)


    # plot loss curve 绘制损失函数曲线
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    # plt.xlabel('epoch')
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    plt.savefig('C:\\Users\\xiaobingnan0224\\Desktop\\fed_mnist_CNN_non-iid_epochs50.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing 在测试集上进行测试
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

