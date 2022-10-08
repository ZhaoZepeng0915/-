import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


# 定义 LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()  # 继承初始化方法时

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.forwardCalculation = nn.Linear(hidden_size, output_size)  # 线性变换

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size);x: <class 'torch.Tensor'>
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size);s, b, h =20,5,16

        x = x.view(s * b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x


if __name__ == '__main__':
    # create database
    data_len = 200
    t = np.linspace(0, 12 * np.pi, data_len)
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    dataset = np.zeros((data_len, 2))
    dataset[:, 0] = sin_t
    dataset[:, 1] = cos_t
    dataset = dataset.astype('float32')

    # fig1，共有200个点序列，这里只显示前60个
    plt.figure()
    plt.plot(t[0:60], dataset[0:60, 0], label='sin(t)')
    plt.plot(t[0:60], dataset[0:60, 1], label='cos(t)')
    plt.plot([2.5, 2.5], [-1.3, 0.55], 'r--', label='t = 2.5')  # t = 2.5
    plt.plot([6.8, 6.8], [-1.3, 0.85], 'm--', label='t = 6.8')  # t = 6.8
    plt.xlabel('t')
    plt.ylim(-1.2, 1.2)
    plt.ylabel('sin(t) and cos(t)')
    plt.legend(loc='upper right')

    # 选择训练集数据
    train_data_ratio = 0.3  # 选前80%的数据作为训练集
    train_data_len = int(data_len * train_data_ratio)
    train_x = dataset[:train_data_len, 0]
    train_y = dataset[:train_data_len, 1]
    INPUT_FEATURES_NUM = 1  # 全局变量
    OUTPUT_FEATURES_NUM = 1
    t_for_training = t[:train_data_len]

    # 剩余数据作为测试集
    test_x = dataset[train_data_len:, 0]
    test_y = dataset[train_data_len:, 1]
    t_for_testing = t[train_data_len:]

    # ----------------- 训练 -------------------
    train_x_tensor = train_x.reshape(-1, 5, INPUT_FEATURES_NUM)  # set batch size to 5，固定五列，行数系统自己计算,n*5*1
    train_y_tensor = train_y.reshape(-1, 5, OUTPUT_FEATURES_NUM)  # set batch size to 5

    # transfer data to pytorch tensor
    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)

    # 实例化类LstmRNN，隐藏层有16个神经元
    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 16, output_size=OUTPUT_FEATURES_NUM, num_layers=1)
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

    max_epochs = 10000
    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor)
        loss = loss_function(output, train_y_tensor) # 计算loss

        loss.backward()  # 反向传播，计算当前梯度
        optimizer.step()  # 根据梯度更新网络参数
        optimizer.zero_grad()  # 清空过往梯度

        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    # 基于训练数据的预测
    predictive_y_for_training = lstm_model(train_x_tensor)
    predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # torch.save(lstm_model.state_dict(), 'model_params.pkl') # 保存模型参数的文件

    # ----------------- 测试 -------------------
    # lstm_model.load_state_dict(torch.load('model_params.pkl'))  # 从文件中下载模型参数

    lstm_model = lstm_model.eval()  # 转换到预测模式

    # 基于测试集进行预测
    test_x_tensor = test_x.reshape(-1, 5, INPUT_FEATURES_NUM)  # set batch size to 5, 和训练时一致
    test_x_tensor = torch.from_numpy(test_x_tensor)

    predictive_y_for_testing = lstm_model(test_x_tensor)
    predictive_y_for_testing = predictive_y_for_testing.view(-1, OUTPUT_FEATURES_NUM).data.numpy()

    # ----------------- plot -------------------
    plt.figure()
    plt.plot(t_for_training, train_x, 'g', label='sin_trn')
    plt.plot(t_for_training, train_y, 'b', label='ref_cos_trn')
    plt.plot(t_for_training, predictive_y_for_training, 'y--', label='pre_cos_trn')

    plt.plot(t_for_testing, test_x, 'c', label='sin_tst')
    plt.plot(t_for_testing, test_y, 'k', label='ref_cos_tst')
    plt.plot(t_for_testing, predictive_y_for_testing, 'm--', label='pre_cos_tst')

    plt.plot([t[train_data_len], t[train_data_len]], [-1.2, 4.0], 'r--', label='separation line')  # separation line

    plt.xlabel('t')
    plt.ylabel('sin(t) and cos(t)')
    plt.xlim(t[0], t[-1])
    plt.ylim(-1.2, 4)
    plt.legend(loc='upper right')
    plt.text(14, 2, "train", size=15, alpha=1.0)
    plt.text(20, 2, "test", size=15, alpha=1.0)

    plt.show()