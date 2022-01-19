# FL_gradient_based

以FedAvg源码为基础，利用channel inversion（引入噪声），上行传梯度，PS进行GD，下行传权值

需改动的主要为已上传的文件

默认选择mnist进行训练验证

进行测试时C-fraction应为1（每轮全部worker都参加），同时应为--all clients （shell命令详见options.py）
